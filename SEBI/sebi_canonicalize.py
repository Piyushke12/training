#!/usr/bin/env python3
"""
SEBI Canonicalization Pipeline

Converts cleaned SEBI legal documents (SEBI_Clean/) into a canonical
XML-tagged JSON format optimized for Continued Pretraining (CPT) of an
8B language model with 128K context.

Produces:
  SEBI_Canonical/<category>/<filename>.json   — per-doc canonical output
  SEBI_Canonical/<category>/_index.json       — per-category index
  SEBI_Canonical/pipeline_report.json         — global validation + stats

Per-doc schema:
  document_id, document_type, title, issued_date, last_amended_date,
  domain, references[], text, structured_text, segments[],
  tables[], structure_summary, validation, cleaning
"""

import os
import re
import sys
import json
import hashlib
import argparse
import statistics
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

SOURCE_ROOT = Path(__file__).parent / "SEBI_Clean"
OUTPUT_ROOT = Path(__file__).parent / "SEBI_Canonical"

MAX_TOKENS_DEFAULT = 24000
CHARS_PER_TOKEN = 3.75

# ---------------------------------------------------------------------------
# Regex library
# ---------------------------------------------------------------------------

# Gazette header — must be EXCLUDED from structural detection
RE_GAZETTE_HEADER = re.compile(r'PART\s+[IVXL]+\s*[-–]\s*SECTION\s+\d+', re.IGNORECASE)

# Structure: chapters (3 forms across categories)
RE_CHAPTER_HYPHEN = re.compile(r'CHAPTER\s+([IVXLCDM]+)\s*[-–]\s+([A-Z][A-Z\s,\'\-]+?)(?=\s|$)', re.IGNORECASE)
RE_CHAPTER_SPACE  = re.compile(r'CHAPTER\s+([IVXLCDM]+)\s+([A-Z][A-Z\s,\'\-]+?)(?=\s|$)', re.IGNORECASE)
RE_CHAPTER_COLON  = re.compile(r'CHAPTER\s+(\d+)\s*:\s+([A-Z][A-Za-z\s,()\-]+?)(?=\s|$)', re.IGNORECASE)

# Structure: regulations (numbered like "1. Short title" or "1. (1) Short title")
# Acts use the same pattern but with "section" semantics; we'll detect generically
RE_NUMBERED_UNIT = re.compile(
    r'(?:^|\n)\s*(\d+[A-Z]?)\s*\.\s+(.+?)(?=\n\s*\d+[A-Z]?\s*\.|\Z)',
    re.DOTALL,
)

# Structure: schedules
RE_SCHEDULE = re.compile(r'SCHEDULE\s*[-–]?\s*([IVXLCDM]+(?:\s*[A-Z])?)\b', re.IGNORECASE)
RE_SCHEDULE_BRACKET = re.compile(r'SCHEDULE\s*[-–]?\s*([IVXLCDM]+(?:\s*[A-Z])?)\s*(?:\[([^\]]+)\])?', re.IGNORECASE)

# Structure: parts
RE_PART = re.compile(r'(?:^|\n)\s*PART\s+([IVXLCDM]+)\b', re.IGNORECASE)

# Structure: forms
RE_FORM = re.compile(r'FORM\s+([A-Z]+(?:\s*\d+)?)\s*(?:\(\s*see\s+(?:Rule|Regulation)\s+\d+\s*\))?', re.IGNORECASE)

# Structure: annexures (Master Circulars)
RE_ANNEXURE = re.compile(r'ANNEXURE\s+([A-Z0-9]+)\b', re.IGNORECASE)

# Structure: Master Circular clauses (1.1, 1.1.2, 15.4.1.1)
RE_MC_CLAUSE = re.compile(r'(?:^|\n)\s*(\d+(?:\.\d+){1,4})\s*\.?\s+(?=[A-Z(])')

# Structure: Guidelines hybrid clauses (1(1.1), 2(ia))
RE_GL_CLAUSE = re.compile(r'(?:^|\n)\s*(\d+)\s*\(\s*([0-9a-z]+\.?[0-9a-z]*)\s*\)\s+(?=[A-Z(])', re.IGNORECASE)

# Structure: General Orders semantic heuristics
RE_GO_WHEREAS = re.compile(r'(?:^|\n)\s*WHEREAS[:\s]', re.IGNORECASE)
RE_GO_NOW_THEREFORE = re.compile(r'(?:^|\n)\s*NOW\s+THEREFORE[:\s]', re.IGNORECASE)
RE_GO_ADJUDICATING = re.compile(r'(?:^|\n)\s*(?:ADJUDICATING\s+OFFICER|Whole\s+Time\s+Member|MEMBER)\s', re.IGNORECASE)
RE_GO_ORDER_NO = re.compile(r'(?:General\s+Order|Order)\s+No\.?\s*(\d+\s+of\s+\d{4})', re.IGNORECASE)

# References
REF_PATTERNS = [
    # (regex, ref_type, needs_normalization)
    (re.compile(r'\bRegulation\s+(\d+[A-Z]?(?:\s*\(\d+\))*(?:\s*\([a-z]\))*)', re.IGNORECASE), 'regulation', True),
    (re.compile(r'\bReg\.\s*(\d+(?:\s*\(\d+\))*(?:\s*\([a-z]\))*)', re.IGNORECASE), 'regulation', True),
    (re.compile(r'\bSection\s+(\d+[A-Z]?(?:\s*\(\d+\))*(?:\s*\([a-z]\))*)', re.IGNORECASE), 'section', True),
    (re.compile(r'\bRule\s+(\d+[A-Z]?(?:\s*\(\d+\))*(?:\s*\([a-z]\))*)', re.IGNORECASE), 'rule', True),
    (re.compile(r'\bsub-regulation\s+\((\d+)\)\s+of\s+(?:regulation\s+)?(\d+[A-Z]?)', re.IGNORECASE), 'subregulation', True),
    (re.compile(r'SEBI/HO/[A-Z0-9/\-]+/CIR/\d{4}/\d+'), 'circular', False),
    (re.compile(r'SEBI/IMD/[A-Z0-9/\-]+/CIR/\d{4}/\d+'), 'circular', False),
    (re.compile(r'\bSEBI\s+Act,?\s+(\d{4})', re.IGNORECASE), 'act', True),
    (re.compile(r'\bSecurities\s+Contracts\s*\(Regulation\)\s*Act,?\s+(\d{4})', re.IGNORECASE), 'act', True),
    (re.compile(r'\bDepositories\s+Act,?\s+(\d{4})', re.IGNORECASE), 'act', True),
    (re.compile(r'\bCompanies\s+Act,?\s+(\d{4})', re.IGNORECASE), 'act', True),
]

# Short-name inline definition (e.g., "hereinafter referred to as 'PFUTP Regulations'")
RE_SHORT_NAME_DEF = re.compile(
    r'(?:hereinafter\s+referred\s+to\s+as|hereinafter\s+called)\s+[“"\']([^"”\']{2,80})[”"\']',
    re.IGNORECASE,
)

# Known short-name → full canonical (corpus-wide map)
SHORT_NAME_CANONICAL = {
    'pfutp regulations': 'SEBI (Prohibition of Fraudulent and Unfair Trade Practices relating to Securities Market) Regulations, 2003',
    'pfutp': 'SEBI (Prohibition of Fraudulent and Unfair Trade Practices relating to Securities Market) Regulations, 2003',
    'sast regulations': 'Securities and Exchange Board of India (Substantial Acquisition of Shares and Takeovers) Regulations, 2011',
    'sast': 'Securities and Exchange Board of India (Substantial Acquisition of Shares and Takeovers) Regulations, 2011',
    'icdr regulations': 'Securities and Exchange Board of India (Issue of Capital and Disclosure Requirements) Regulations, 2018',
    'icdr': 'Securities and Exchange Board of India (Issue of Capital and Disclosure Requirements) Regulations, 2018',
    'lodr regulations': 'Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015',
    'lodr': 'Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015',
    'secc regulations': 'Securities and Exchange Board of India (Issue and Listing of Securitised Debt Instruments and Security Receipts) Regulations, 2008',
    'secc': 'Securities and Exchange Board of India (Issue and Listing of Securitised Debt Instruments and Security Receipts) Regulations, 2008',
    'aif regulations': 'Securities and Exchange Board of India (Alternative Investment Funds) Regulations, 2012',
    'aif': 'Securities and Exchange Board of India (Alternative Investment Funds) Regulations, 2012',
    'fpi regulations': 'Securities and Exchange Board of India (Foreign Portfolio Investors) Regulations, 2019',
    'fpi': 'Securities and Exchange Board of India (Foreign Portfolio Investors) Regulations, 2019',
    'mf regulations': 'Securities and Exchange Board of India (Mutual Funds) Regulations, 1996',
    'mf': 'Securities and Exchange Board of India (Mutual Funds) Regulations, 1996',
    'takeover regulations': 'Securities and Exchange Board of India (Substantial Acquisition of Shares and Takeovers) Regulations, 2011',
    'takeover code': 'Securities and Exchange Board of India (Substantial Acquisition of Shares and Takeovers) Regulations, 2011',
    'buy-back regulations': 'Securities and Exchange Board of India (Buy-back of Securities) Regulations, 2018',
    'buyback regulations': 'Securities and Exchange Board of India (Buy-back of Securities) Regulations, 2018',
}

# Domain classification — keyword → domain
DOMAIN_KEYWORDS = [
    ('LODR', ['listing obligations', 'lodr', 'listing agreement', 'stock exchange listing']),
    ('AIF', ['alternative investment', 'aif', 'venture capital', 'private equity', 'hedge fund']),
    ('Mutual Funds', ['mutual fund', 'mf scheme', 'asset management', 'amc']),
    ('FPI', ['foreign portfolio investor', 'fpi', 'foreign institutional', 'fii']),
    ('Insider Trading', ['insider trading', 'insider']),
    ('PFUTP', ['fraudulent and unfair', 'pfutp', 'unfair trade']),
    ('Merchant Banking', ['merchant bank']),
    ('Investment Advisers', ['investment adviser', 'investment advisor']),
    ('Credit Rating Agencies', ['credit rating', 'cra ']),
    ('Takeover Code', ['takeover', 'substantial acquisition', 'sast']),
    ('Depositories', ['depositor', 'demat', 'depository participant', 'dp ']),
    ('Buy-back', ['buy-back', 'buyback']),
    ('ICDR', ['issue of capital', 'disclosure requirement', 'icdr', 'public issue', 'rights issue']),
    ('General Securities Market', ['stock exchange', 'clearing corporation', 'securities', 'trading']),
]

# Date patterns (verified across categories)
DATE_PATTERNS = [
    re.compile(r'\[?Last\s+amend(?:ed|ment)\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\]?', re.IGNORECASE),
    re.compile(r'(?:Amended|updated)\s+upto\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
    re.compile(r'\bas\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
    re.compile(r'dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4})', re.IGNORECASE),
    re.compile(r'\b(?:w\.?e\.?f\.?\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
]

MONTH_NAMES = {m.lower(): i+1 for i, m in enumerate([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars/3.75."""
    return int(len(text) / CHARS_PER_TOKEN)


def slugify(s: str, max_len: int = 60) -> str:
    """Filesystem-safe slug."""
    s = re.sub(r'[^\w\s-]', '', s).strip().lower()
    s = re.sub(r'[\s_-]+', '-', s)
    return s[:max_len].strip('-')


def sha8(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:8]


def build_document_id(category: str, year: str, source_file: str) -> str:
    cat_slug = slugify(category, 30)
    yr = year or '0000'
    return f"{cat_slug}/{yr}/{sha8(source_file)}"


def normalize_whitespace(text: str) -> str:
    """Normalize smart quotes, devanagari residue, and excessive whitespace."""
    if not text:
        return text
    # Smart quotes → ASCII
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    # Strip any residual Devanagari (safety net)
    text = re.sub(r'[ऀ-ॿ꣠-ꣿ᳐-᳿]+', '', text)
    # Normalize unicode whitespace to ASCII space
    text = unicodedata.normalize('NFKC', text)
    # Collapse runs of spaces/tabs (preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip trailing spaces per line
    text = re.sub(r' *\n', '\n', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fix_broken_tokens(text: str) -> str:
    """Rejoin tokens broken across PDF lines: 'Regulation\\n283' → 'Regulation 283'."""
    # Word + newline + number (structural markers)
    text = re.sub(
        r'\b(Regulation|Section|Rule|Schedule|Chapter|Part|Form|Annexure|Clause)\s*\n\s*(\d+[A-Z]?)',
        r'\1 \2', text, flags=re.IGNORECASE
    )
    # Circular numbers split across lines
    text = re.sub(r'(SEBI/[A-Z]+)\s*\n\s*([A-Z0-9/\-]+)', r'\1\2', text)
    # "Section 11" where "11" continues on next line — already handled above
    # Sub-regulation parens split: "30\n(1)" → "30(1)"
    text = re.sub(r'(\d+[A-Z]?)\s*\n\s*\(', r'\1(', text)
    return text


def parse_iso_date(date_str: str) -> str:
    """Parse a date string and return ISO format YYYY-MM-DD, or YYYY-00-00 if only year."""
    if not date_str:
        return None
    s = date_str.strip()
    # Try full date: "March 21, 2026" or "21 March 2026" or "21st March, 2026"
    m = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+),?\s+(\d{4})', s, re.I)
    if m:
        day, month, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if month in MONTH_NAMES:
            return f"{year:04d}-{MONTH_NAMES[month]:02d}-{day:02d}"
    m = re.match(r'([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})', s, re.I)
    if m:
        month, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        if month in MONTH_NAMES:
            return f"{year:04d}-{MONTH_NAMES[month]:02d}-{day:02d}"
    # Year only
    m = re.match(r'^(\d{4})$', s)
    if m:
        return f"{m.group(1)}-00-00"
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Block:
    name: str  # 'chapter', 'regulation', 'section', 'schedule', 'rule', 'clause', etc.
    number: str = ''
    title: str = ''
    body: str = ''
    page: int = 0
    block_id: str = ''
    children: list = field(default_factory=list)


@dataclass
class Reference:
    type: str
    raw_text: str
    canonical: str
    target_document_id: str = None
    target_match_confidence: float = None
    occurrences: int = 1


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def parse_dates(title: str, first_pages_text: str, year_meta: str) -> dict:
    """Extract issued_date and last_amended_date.

    issued_date priority (for Regulations/Rules/Acts):
      1. "dated <date>" pattern (e.g., "dated January 20, 1993")
      2. year_meta fallback
    issued_date priority (for Master Circulars):
      1. "as on <date>" pattern (e.g., "as on March 20, 2026")
      2. year_meta fallback
    last_amended_date priority:
      1. "Last amended on <date>"
      2. "Amended upto <date>" / "updated upto <date>"
    """
    result = {"issued_date": None, "last_amended_date": None}
    search_text = f"{title} {first_pages_text[:3000]}"

    # Last amended date: "Last amended on" or "Amended upto" / "updated upto"
    for pat in DATE_PATTERNS[:2]:
        m = pat.search(search_text)
        if m:
            iso = parse_iso_date(m.group(1))
            if iso:
                result["last_amended_date"] = iso
                break

    # Issued date: prefer "dated" (Rules/Regs), then "as on" (Master Circulars)
    dated_pat = re.compile(r'dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4})', re.IGNORECASE)
    as_on_pat = re.compile(r'\bas\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE)

    m = dated_pat.search(search_text)
    if m:
        iso = parse_iso_date(m.group(1))
        if iso:
            result["issued_date"] = iso
    if not result["issued_date"]:
        m = as_on_pat.search(search_text)
        if m:
            iso = parse_iso_date(m.group(1))
            if iso:
                result["issued_date"] = iso

    # Year-only fallback (used when nothing more specific found)
    if not result["issued_date"] and year_meta:
        result["issued_date"] = f"{year_meta}-00-00"

    return result


def classify_domain(title: str, text_sample: str) -> str:
    """Rule-based domain classification.

    Title is the primary signal — body text is only consulted when the title
    is ambiguous. This prevents docs like the SEBI Act (which mentions "mutual
    funds" in passing) from being misclassified as Mutual Funds.
    """
    title_lower = title.lower()
    # First pass: title only
    for domain, keywords in DOMAIN_KEYWORDS:
        for kw in keywords:
            if kw in title_lower:
                return domain
    # Second pass: title + first 2K of body, but require a strong signal
    # (2+ distinct keywords from the same domain)
    sample = f"{title} {text_sample[:2000]}".lower()
    for domain, keywords in DOMAIN_KEYWORDS:
        matches = sum(1 for kw in keywords if kw in sample)
        if matches >= 2:
            return domain
    return "General Securities Market"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def join_pages(pages: list) -> str:
    """Concatenate page texts with double-newline separators."""
    parts = []
    for p in pages:
        t = p.get("text", "") or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Structure detection — per category
# ---------------------------------------------------------------------------

def detect_structure_Regulations(text: str) -> list:
    """Detect chapter/regulation/schedule structure."""
    blocks = []
    # Strategy: walk text linearly, find headers in order
    # We'll use a combined regex with alternation, then emit blocks between matches
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<chapter>CHAPTER\s+[IVXLCDM]+\s*[-–]\s+[A-Z][A-Z\s,\'\-]+)'
        r'|(?P<schedule>SCHEDULE\s*[-–]?\s*[IVXLCDM]+(?:\s*[A-Z])?(?:\s*\[[^\]]+\])?)'
        r'|(?P<part>PART\s+[IVXLCDM]+)'
        r'|(?P<regulation>(?:^|\n)\s*\d+[A-Z]?\s*\.\s+(?=[A-Z(]))'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    # Capture any preamble before first match
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('chapter'):
            # Parse "CHAPTER I - PRELIMINARY"
            cm = re.match(r'CHAPTER\s+([IVXLCDM]+)\s*[-–]\s*(.+)', m.group('chapter'), re.I)
            num = cm.group(1) if cm else ''
            title = cm.group(2).strip() if cm else ''
            blocks.append(Block(name='chapter', number=num, title=title, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('schedule'):
            sm = re.match(r'SCHEDULE\s*[-–]?\s*([IVXLCDM]+(?:\s*[A-Z])?)', m.group('schedule'), re.I)
            num = sm.group(1).strip() if sm else ''
            blocks.append(Block(name='schedule', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('part'):
            pm = re.match(r'PART\s+([IVXLCDM]+)', m.group('part'), re.I)
            num = pm.group(1) if pm else ''
            blocks.append(Block(name='part', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('regulation'):
            rm = re.match(r'\s*(\d+[A-Z]?)\s*\.\s+(.+)', m.group('regulation').strip(), re.I)
            num = rm.group(1) if rm else ''
            # Title is first line of body
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='regulation', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


def detect_structure_Acts(text: str) -> list:
    """Acts use Section instead of Regulation."""
    blocks = []
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<chapter>CHAPTER\s+[IVXLCDM]+\s*[-–]?\s*[A-Z][A-Z\s,\'\-]+)'
        r'|(?P<schedule>SCHEDULE\s*[-–]?\s*[IVXLCDM]+(?:\s*[A-Z])?(?:\s*\[[^\]]+\])?)'
        r'|(?P<section>(?:^|\n)\s*\d+[A-Z]?\s*\.\s+(?=[A-Z(]))'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('chapter'):
            cm = re.match(r'CHAPTER\s+([IVXLCDM]+)\s*[-–]?\s*(.+)?', m.group('chapter'), re.I)
            num = cm.group(1) if cm else ''
            title = (cm.group(2) or '').strip() if cm else ''
            blocks.append(Block(name='chapter', number=num, title=title, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('schedule'):
            sm = re.match(r'SCHEDULE\s*[-–]?\s*([IVXLCDM]+(?:\s*[A-Z])?)', m.group('schedule'), re.I)
            num = sm.group(1).strip() if sm else ''
            blocks.append(Block(name='schedule', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('section'):
            rm = re.match(r'\s*(\d+[A-Z]?)\s*\.\s+(.+)', m.group('section').strip(), re.I)
            num = rm.group(1) if rm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='section', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


def detect_structure_Rules(text: str) -> list:
    """Rules: flat structure with Rule N."""
    blocks = []
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<schedule>SCHEDULE\s*[-–]?\s*[IVXLCDM]+(?:\s*[A-Z])?(?:\s*\[[^\]]+\])?)'
        r'|(?P<form>FORM\s+[A-Z]+(?:\s*\(\s*see\s+Rule\s+\d+\s*\))?)'
        r'|(?P<rule>(?:^|\n)\s*\d+[A-Z]?\s*\.\s+(?=[A-Z(]))'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('schedule'):
            sm = re.match(r'SCHEDULE\s*[-–]?\s*([IVXLCDM]+(?:\s*[A-Z])?)', m.group('schedule'), re.I)
            num = sm.group(1).strip() if sm else ''
            blocks.append(Block(name='schedule', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('form'):
            fm = re.match(r'FORM\s+([A-Z]+)', m.group('form'), re.I)
            num = fm.group(1) if fm else ''
            blocks.append(Block(name='form', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('rule'):
            rm = re.match(r'\s*(\d+[A-Z]?)\s*\.\s+(.+)', m.group('rule').strip(), re.I)
            num = rm.group(1) if rm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='rule', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


def detect_structure_MasterCirculars(text: str) -> list:
    """Master Circulars: numbered chapters + numbered clauses (1.1, 1.1.2).

    Also accepts bare numbered paragraphs (1., 2., 17.) for older circulars
    that don't use the period-delimited clause format.
    """
    blocks = []
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<chapter>CHAPTER\s+\d+\s*:\s+[A-Z][A-Za-z\s,()\-]+)'
        r'|(?P<chapter_roman>CHAPTER\s+[IVXLCDM]+\s*[-–]?\s+[A-Z][A-Z\s,\'\-]+)'
        r'|(?P<annexure>ANNEXURE\s+[A-Z0-9]+)'
        r'|(?P<clause>(?:^|\n)\s*\d+(?:\.\d+){1,4}\s*\.?\s+(?=[A-Z(]))'
        r'|(?P<para>(?:^|\n)\s*\d+\.\s+(?=[A-Z(]))'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('chapter'):
            cm = re.match(r'CHAPTER\s+(\d+)\s*:\s+(.+)', m.group('chapter'), re.I)
            num = cm.group(1) if cm else ''
            title = cm.group(2).strip() if cm else ''
            blocks.append(Block(name='chapter', number=num, title=title, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('chapter_roman'):
            cm = re.match(r'CHAPTER\s+([IVXLCDM]+)\s*[-–]?\s*(.+)?', m.group('chapter_roman'), re.I)
            num = cm.group(1) if cm else ''
            title = (cm.group(2) or '').strip() if cm else ''
            blocks.append(Block(name='chapter', number=num, title=title, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('annexure'):
            am = re.match(r'ANNEXURE\s+([A-Z0-9]+)', m.group('annexure'), re.I)
            num = am.group(1) if am else ''
            blocks.append(Block(name='annexure', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('clause'):
            cm = re.match(r'\s*(\d+(?:\.\d+){1,4})\s*\.?\s+(.+)?', m.group('clause').strip(), re.I)
            num = cm.group(1) if cm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='clause', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('para'):
            cm = re.match(r'\s*(\d+)\.\s+', m.group('para').strip(), re.I)
            num = cm.group(1) if cm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='clause', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


def detect_structure_Guidelines(text: str) -> list:
    """Guidelines: hybrid CHAPTER I + 1(1.1) clauses + N.M clauses + Schedule + bare N."""
    blocks = []
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<chapter>CHAPTER\s+[IVXLCDM]+\s*[-–]?\s*[A-Z][A-Z\s,\'\-]+)'
        r'|(?P<schedule>Schedule\s+[IVXLCDM]+(?:\s*[A-Z])?(?:\s*\[[^\]]+\])?)'
        r'|(?P<clause>(?:^|\n)\s*\d+\s*\(\s*[0-9a-z]+\.?[0-9a-z]*\s*\)\s+(?=[A-Z(]))'
        r'|(?P<sub_clause>(?:^|\n)\s*\d+\.\d+\s+(?=[A-Z(]))'
        r'|(?P<para>(?:^|\n)\s*\d+\.\s+(?=[A-Z(]))'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('chapter'):
            cm = re.match(r'CHAPTER\s+([IVXLCDM]+)\s*[-–]?\s*(.+)?', m.group('chapter'), re.I)
            num = cm.group(1) if cm else ''
            title = (cm.group(2) or '').strip() if cm else ''
            blocks.append(Block(name='chapter', number=num, title=title, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('schedule'):
            sm = re.match(r'Schedule\s+([IVXLCDM]+(?:\s*[A-Z])?)', m.group('schedule'), re.I)
            num = sm.group(1).strip() if sm else ''
            blocks.append(Block(name='schedule', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('clause'):
            cm = re.match(r'\s*(\d+)\s*\(\s*([0-9a-z]+\.?[0-9a-z]*)\s*\)', m.group('clause').strip(), re.I)
            num = f"{cm.group(1)}({cm.group(2)})" if cm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='clause', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('sub_clause'):
            cm = re.match(r'\s*(\d+\.\d+)\s+', m.group('sub_clause').strip(), re.I)
            num = cm.group(1) if cm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='clause', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('para'):
            cm = re.match(r'\s*(\d+)\.\s+', m.group('para').strip(), re.I)
            num = cm.group(1) if cm else ''
            first_line = body.split('\n', 1)[0].strip()[:120] if body else ''
            blocks.append(Block(name='clause', number=num, title=first_line, body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


def detect_structure_GeneralOrders(text: str) -> list:
    """General Orders: best-effort semantic roles via heuristics."""
    blocks = []
    # Order headers
    HEADER_RE = re.compile(
        r'(?:'
        r'(?P<order_no>Order\s+No\.?\s*\d+\s+of\s+\d{4})'
        r'|(?P<whereas>WHEREAS[:\s])'
        r'|(?P<now_therefore>NOW\s+THEREFORE[:\s])'
        r'|(?P<adjudicating>(?:ADJUDICATING\s+OFFICER|Whole\s+Time\s+Member|MEMBER)\s)'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Block(name='text', body=text, block_id='blk_0')]

    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            blocks.append(Block(name='preamble', body=pre, block_id='blk_0'))

    for i, m in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if m.group('order_no'):
            om = re.match(r'Order\s+No\.?\s*(\d+\s+of\s+\d{4})', m.group('order_no'), re.I)
            num = om.group(1) if om else ''
            blocks.append(Block(name='order_header', number=num, body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('whereas'):
            blocks.append(Block(name='recital', title='facts', body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('now_therefore'):
            blocks.append(Block(name='direction', title='directions', body=body, block_id=f'blk_{len(blocks)}'))
        elif m.group('adjudicating'):
            blocks.append(Block(name='penalty', title='penalty', body=body, block_id=f'blk_{len(blocks)}'))

    return blocks


DETECTORS = {
    'Acts': detect_structure_Acts,
    'Rules': detect_structure_Rules,
    'Regulations': detect_structure_Regulations,
    'Master Circulars': detect_structure_MasterCirculars,
    'Guidelines': detect_structure_Guidelines,
    'General Orders': detect_structure_GeneralOrders,
}

DOC_TYPE_MAP = {
    'Acts': 'Act',
    'Rules': 'Rule',
    'Regulations': 'Regulation',
    'Master Circulars': 'MasterCircular',
    'Guidelines': 'Guideline',
    'General Orders': 'GeneralOrder',
}


# ---------------------------------------------------------------------------
# Table serialization
# ---------------------------------------------------------------------------

def serialize_table(table: list, max_rows: int = 50) -> list:
    """Serialize a table (list of rows) as markdown. Split >max_rows into chunks.

    Each chunk repeats the header row so it's self-contained.
    Returns list of markdown strings (one per chunk).
    """
    if not table or not any(row for row in table):
        return []

    def clean_cell(c):
        if c is None:
            return ''
        return str(c).replace('|', '\\|').replace('\n', ' ').strip()

    def render(rows):
        if not rows:
            return ''
        ncols = max(len(r) for r in rows)
        padded = [list(r) + [''] * (ncols - len(r)) for r in rows]
        md = ['| ' + ' | '.join(clean_cell(c) for c in padded[0]) + ' |',
              '| ' + ' | '.join('---' for _ in padded[0]) + ' |']
        for r in padded[1:]:
            md.append('| ' + ' | '.join(clean_cell(c) for c in r) + ' |')
        return '\n'.join(md)

    if len(table) <= max_rows:
        md = render(table)
        return [md] if md else []

    # Split into chunks of max_rows, each prefixed with the header
    header = table[0]
    out = []
    for i in range(1, len(table), max_rows):
        chunk = [header] + table[i:i+max_rows]
        md = render(chunk)
        if md:
            out.append(md)
    return out


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def normalize_reference(raw: str, ref_type: str) -> str:
    """Normalize a reference string to canonical form."""
    s = raw.strip()
    if ref_type == 'regulation':
        s = re.sub(r'\bReg\.', 'Regulation', s, flags=re.I)
        s = re.sub(r'\s*\)\s*\(', ')(', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    if ref_type == 'section':
        s = re.sub(r'\s*\)\s*\(', ')(', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    if ref_type == 'rule':
        s = re.sub(r'\s*\)\s*\(', ')(', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    if ref_type == 'subregulation':
        m = re.match(r'\((\d+)\)\s+of\s+(?:regulation\s+)?(\d+[A-Z]?)', s, re.I)
        if m:
            return f"Regulation {m.group(2)}({m.group(1)})"
        return s
    if ref_type == 'act':
        m = re.search(r'(\d{4})', s)
        if m:
            if 'SEBI Act' in s or 'sebi act' in s:
                return f"SEBI Act, {m.group(1)}"
            if 'Depositories' in s or 'depositories' in s:
                return f"Depositories Act, {m.group(1)}"
            if 'Companies' in s or 'companies' in s:
                return f"Companies Act, {m.group(1)}"
            if 'Securities Contracts' in s or 'securities contracts' in s:
                return f"SCRA Act, {m.group(1)}"
        return s
    return s


def detect_short_names(text: str) -> list:
    """Detect inline short-name definitions in first ~10 pages worth of text."""
    sample = text[:30000]  # first ~10 pages
    out = []
    for m in RE_SHORT_NAME_DEF.finditer(sample):
        short_name = m.group(1).strip()
        context_start = max(0, m.start() - 200)
        context_end = min(len(text), m.end() + 200)
        definition_text = sample[context_start:context_end].strip()
        full_name = SHORT_NAME_CANONICAL.get(short_name.lower())
        out.append({
            "short_name": short_name,
            "definition_text": definition_text,
            "full_name_resolved": full_name,
        })
    # Dedupe by short_name
    seen = set()
    unique = []
    for s in out:
        key = s["short_name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def extract_references(text: str, short_names: list) -> list:
    """Extract all references from text. Returns list of Reference objects."""
    refs = []
    for rx, ref_type, needs_norm in REF_PATTERNS:
        for m in rx.finditer(text):
            raw = m.group(0)
            canonical = normalize_reference(raw, ref_type) if needs_norm else raw
            refs.append(Reference(
                type=ref_type,
                raw_text=raw,
                canonical=canonical,
            ))
    # Dedupe by (type, canonical), count occurrences
    deduped = {}
    for r in refs:
        key = (r.type, r.canonical)
        if key in deduped:
            deduped[key].occurrences += 1
        else:
            deduped[key] = r
    return list(deduped.values())


def resolve_references(refs: list, doc_index: dict, current_doc_id: str) -> list:
    """Resolve references to corpus documents.

    Strategy:
      - 'act' references: substring-match against titles. E.g. canonical
        'SEBI Act, 1992' matches any title containing 'SEBI Act' and '1992'.
        Confidence = 1.0 for exact substring, 0.9 for fuzzy.
      - 'circular' references: match circular number against any title or
        source_file (circular numbers are sometimes in filenames).
      - 'regulation'/'section'/'rule'/'subregulation': cannot resolve to a
        specific doc without knowing the parent act/regulation. Left null
        (honest — we don't know which regulation 'Regulation 30' refers to
        without more context).
    """
    for r in refs:
        if r.type in ('regulation', 'section', 'rule', 'subregulation'):
            r.target_document_id = None
            continue

        canonical_lower = r.canonical.lower()

        if r.type == 'act':
            # Map common short forms to full names for matching
            ACT_NAME_ALIASES = {
                'sebi act': 'securities and exchange board of india act',
                'scra act': 'securities contracts (regulation) act',
                'depositories act': 'depositories act',
                'companies act': 'companies act',
            }
            m = re.match(r'(.+Act),?\s+(\d{4})', r.canonical, re.I)
            if not m:
                r.target_document_id = None
                continue
            act_short = m.group(1).strip().lower()
            act_year = m.group(2)
            act_full = ACT_NAME_ALIASES.get(act_short, act_short)

            best_id = None
            best_conf = 0.0
            for doc_id, info in doc_index.items():
                if doc_id == current_doc_id:
                    continue
                title_lower = info["title"].lower()
                if act_full in title_lower and act_year in title_lower:
                    # Confidence based on how much of the title is the act name
                    conf = 1.0 - (len(title_lower) - len(act_full)) / max(1, len(title_lower))
                    conf = max(0.75, min(0.99, conf))
                    if conf > best_conf:
                        best_conf = conf
                        best_id = doc_id
            r.target_document_id = best_id
            r.target_match_confidence = round(best_conf, 3) if best_id else None
            continue

        if r.type == 'circular':
            # Match circular number against titles and source files
            best_id = None
            best_conf = 0.0
            for doc_id, info in doc_index.items():
                if doc_id == current_doc_id:
                    continue
                title_lower = info["title"].lower()
                if canonical_lower in title_lower:
                    conf = 0.95
                    if conf > best_conf:
                        best_conf = conf
                        best_id = doc_id
            r.target_document_id = best_id
            r.target_match_confidence = round(best_conf, 3) if best_id else None
            continue

        r.target_document_id = None
    return refs


# ---------------------------------------------------------------------------
# structured_text builder
# ---------------------------------------------------------------------------

def escape_xml(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;'))


def render_block_xml(block: Block) -> str:
    """Render a single block as XML."""
    attrs = []
    if block.number:
        attrs.append(f'number="{escape_xml(block.number)}"')
    if block.title:
        attrs.append(f'title="{escape_xml(block.title[:120])}"')
    attr_str = (' ' + ' '.join(attrs)) if attrs else ''
    body = escape_xml(block.body)
    return f"<{block.name}{attr_str}>\n{body}\n</{block.name}>"


def build_structured_text(meta: dict, blocks: list, tables: list) -> str:
    """Build the full XML-tagged structured_text.

    Tables are rendered in a <tables> section at the end of <body>, each
    wrapped in a <table> tag with id/page attributes. Inline embedding per-
    block would require page-level position tracking not currently available
    from the block stream.
    """
    parts = []
    parts.append(f'<document id="{escape_xml(meta["document_id"])}">')
    parts.append(f'  <document_type>{escape_xml(meta["document_type"])}</document_type>')
    parts.append(f'  <title>{escape_xml(meta["title"])}</title>')
    if meta.get("issued_date"):
        parts.append(f'  <issued_date>{escape_xml(meta["issued_date"])}</issued_date>')
    if meta.get("last_amended_date"):
        parts.append(f'  <last_amended_date>{escape_xml(meta["last_amended_date"])}</last_amended_date>')
    parts.append(f'  <domain>{escape_xml(meta["domain"])}</domain>')
    # References inline
    if meta.get("references"):
        parts.append('  <references>')
        for r in meta["references"]:
            ref_attrs = [f'type="{escape_xml(r["type"])}"', f'canonical="{escape_xml(r["canonical"])}"']
            if r.get("target_document_id"):
                ref_attrs.append(f'target_document_id="{escape_xml(r["target_document_id"])}"')
            parts.append(f'    <reference {" ".join(ref_attrs)}/>')
        parts.append('  </references>')
    parts.append('  <body>')
    for block in blocks:
        rendered = render_block_xml(block)
        for line in rendered.split('\n'):
            parts.append(f'    {line}')
    # Tables section at end of body
    if tables:
        parts.append('    <tables>')
        for t in tables:
            parts.append(f'      <table id="{escape_xml(t["table_id"])}" page="{t["page"]}" rows="{t["rows"]}" cols="{t["cols"]}">')
            md = t["markdown"]
            for line in md.split('\n'):
                parts.append(f'        {escape_xml(line)}')
            parts.append('      </table>')
        parts.append('    </tables>')
    parts.append('  </body>')
    parts.append('</document>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_blocks(blocks: list, max_tokens: int, meta: dict = None, total_segments_estimate: int = None) -> list:
    """Greedy packing with structural-boundary respect.

    Never splits mid-regulation/mid-section/mid-rule/mid-clause. A single
    oversized block is split at paragraph boundaries as a last resort.

    Each segment's `text` field includes a compact XML preamble (document_id,
    type, title, dates, domain, segment index/total) so segments are
    self-contained for RAG indexing.
    """
    segments = []
    current_blocks = []
    current_tokens = 0

    def flush_current():
        nonlocal current_blocks, current_tokens
        if current_blocks:
            segments.append(flush_segment(current_blocks, len(segments), meta, total_segments_estimate))
            current_blocks = []
            current_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block.body) + 20  # overhead for tags

        if block_tokens > max_tokens:
            # Single oversized block — flush current, then split at paragraph boundaries
            flush_current()
            subs = split_oversized_block(block, max_tokens)
            for i, sub in enumerate(subs):
                preamble = build_segment_preamble(meta, len(segments), total_segments_estimate)
                full_text = f"{preamble}\n    {sub}\n  </body>\n</document>"
                segments.append({
                    "segment_id": f"{block.block_id}_seg_{i}",
                    "segment_index": len(segments),
                    "tokens_estimate": estimate_tokens(full_text),
                    "char_count": len(full_text),
                    "block_ids": [block.block_id],
                    "text": full_text,
                    "oversized": True,
                })
            continue

        if current_tokens + block_tokens > max_tokens and current_blocks:
            flush_current()
        current_blocks.append(block)
        current_tokens += block_tokens

    flush_current()
    return segments


def build_segment_preamble(meta: dict, seg_index: int, total: int = None) -> str:
    """Compact XML preamble repeated on every segment for RAG standalone use."""
    if not meta:
        return ""
    total_str = str(total) if total else "?"
    lines = [
        f'<document id="{escape_xml(meta.get("document_id", ""))}">',
        f'  <document_type>{escape_xml(meta.get("document_type", ""))}</document_type>',
        f'  <title>{escape_xml(meta.get("title", ""))}</title>',
    ]
    if meta.get("issued_date"):
        lines.append(f'  <issued_date>{escape_xml(meta["issued_date"])}</issued_date>')
    if meta.get("last_amended_date"):
        lines.append(f'  <last_amended_date>{escape_xml(meta["last_amended_date"])}</last_amended_date>')
    lines.append(f'  <domain>{escape_xml(meta.get("domain", ""))}</domain>')
    lines.append(f'  <segment index="{seg_index}" total_segments="{total_str}"/>')
    lines.append('  <body>')
    return '\n'.join(lines)


def split_oversized_block(block: Block, max_tokens: int) -> list:
    """Split an oversized block at paragraph boundaries, then sentence boundaries."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    paras = re.split(r'\n{2,}', block.body)

    # First pass: pack paragraphs into parts under max_chars
    parts = []
    current = []
    current_chars = 0
    for para in paras:
        if current_chars + len(para) > max_chars and current:
            parts.append('\n\n'.join(current))
            current = []
            current_chars = 0
        # If the paragraph itself is too big, split at sentence boundaries
        if len(para) > max_chars:
            if current:
                parts.append('\n\n'.join(current))
                current = []
                current_chars = 0
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if current_chars + len(sent) > max_chars and current:
                    parts.append('\n\n'.join(current))
                    current = []
                    current_chars = 0
                current.append(sent)
                current_chars += len(sent)
        else:
            current.append(para)
            current_chars += len(para)
    if current:
        parts.append('\n\n'.join(current))

    return [render_block_xml(Block(name=block.name, number=block.number, title=block.title, body=p, block_id=block.block_id)) for p in parts]


def flush_segment(blocks: list, seg_index: int, meta: dict = None, total: int = None) -> dict:
    """Package a list of blocks into a segment dict with XML preamble."""
    body_parts = [render_block_xml(b) for b in blocks]
    body_content = '\n'.join(body_parts)
    preamble = build_segment_preamble(meta, seg_index, total)
    full_text = f"{preamble}\n    {body_content}\n  </body>\n</document>" if preamble else body_content
    return {
        "segment_id": f"seg_{seg_index}",
        "segment_index": seg_index,
        "tokens_estimate": estimate_tokens(full_text),
        "char_count": len(full_text),
        "block_ids": [b.block_id for b in blocks],
        "text": full_text,
        "oversized": False,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_empty_tag_rate(structured_text: str) -> float:
    """Fraction of tags that are empty."""
    empty_tags = re.findall(r'<(\w+)[^>]*>\s*</\1>', structured_text)
    all_tags = re.findall(r'<(\w+)[^>]*>', structured_text)
    if not all_tags:
        return 0.0
    return len(empty_tags) / len(all_tags)


def compute_structural_coverage(blocks: list, total_text: str) -> float:
    """Fraction of text chars inside structural blocks vs total."""
    if not total_text:
        return 0.0
    in_blocks = sum(len(b.body) for b in blocks if b.name not in ('preamble', 'text'))
    return min(1.0, in_blocks / max(1, len(total_text)))


def validate_document(doc: dict, blocks: list, total_text: str) -> dict:
    seg_tokens = [s["tokens_estimate"] for s in doc.get("segments", [])]
    ref_types = Counter(r["type"] for r in doc.get("references", []))
    warnings = []
    if doc.get("status") == "empty":
        warnings.append("empty_document: 0 pages")
    oversized = sum(1 for s in doc.get("segments", []) if s.get("oversized"))
    if oversized:
        warnings.append(f"{oversized} oversized segment(s) (single block > max_tokens)")
    coverage = compute_structural_coverage(blocks, total_text)
    # Only warn on low coverage for non-empty docs
    if coverage < 0.50 and len(total_text) > 200:
        warnings.append(f"low_structural_coverage: {coverage:.1%}")
    return {
        "empty_tag_rate": round(compute_empty_tag_rate(doc.get("structured_text", "")), 4),
        "structural_coverage": round(coverage, 4),
        "segment_count": len(doc.get("segments", [])),
        "segment_token_distribution": {
            "min": min(seg_tokens) if seg_tokens else 0,
            "p50": int(statistics.median(seg_tokens)) if seg_tokens else 0,
            "p95": int(statistics.quantiles(seg_tokens, n=20)[18]) if len(seg_tokens) > 1 else (seg_tokens[0] if seg_tokens else 0),
            "max": max(seg_tokens) if seg_tokens else 0,
        },
        "oversized_segments": oversized,
        "reference_count": len(doc.get("references", [])),
        "reference_types": dict(ref_types),
        "unresolved_references_rate": round(
            sum(1 for r in doc.get("references", []) if not r.get("target_document_id"))
            / max(1, len(doc.get("references", []))),
            4,
        ),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_one(input_path: Path, doc_index: dict) -> dict:
    """Process a single document."""
    src = json.loads(input_path.read_text())
    category = src.get("category", "")
    year = src.get("year", "")
    title = src.get("title", input_path.stem)
    source_file = src.get("source_file", "")
    pages = src.get("pages", [])
    cleaning = src.get("cleaning", {})

    document_id = build_document_id(category, year, source_file)
    document_type = DOC_TYPE_MAP.get(category, "Unknown")

    # Empty doc handling
    if not pages or src.get("total_chars", 0) == 0:
        result = {
            "document_id": document_id,
            "document_type": document_type,
            "source": {
                "category": category,
                "source_file": source_file,
                "source_urls": src.get("source_urls", {}),
                "input_json_path": str(input_path.resolve().relative_to(SOURCE_ROOT)),
            },
            "title": normalize_whitespace(title),
            "year": year,
            "issued_date": None,
            "last_amended_date": None,
            "status": "empty",
            "domain": classify_domain(title, ""),
            "references": [],
            "short_names_defined": [],
            "text": "",
            "structured_text": f'<document id="{escape_xml(document_id)}">\n  <document_type>{escape_xml(document_type)}</document_type>\n  <title>{escape_xml(title)}</title>\n  <body/>\n</document>',
            "segments": [],
            "tables": [],
            "structure_summary": {"total_blocks": 0, "block_types": {}, "max_depth": 0},
            "validation": {
                "empty_tag_rate": 0.0,
                "structural_coverage": 0.0,
                "segment_count": 0,
                "segment_token_distribution": {"min": 0, "p50": 0, "p95": 0, "max": 0},
                "oversized_segments": 0,
                "reference_count": 0,
                "reference_types": {},
                "unresolved_references_rate": 0.0,
                "warnings": ["empty_document: 0 pages"],
            },
            "cleaning": cleaning,
        }
        return result

    # Normalize text
    raw_text = join_pages(pages)
    text = normalize_whitespace(raw_text)
    text = fix_broken_tokens(text)

    # Metadata
    first_pages = '\n'.join(p.get('text', '') for p in pages[:3])
    dates = parse_dates(title, first_pages, year)
    domain = classify_domain(title, text[:4000])

    # Structure detection
    detector = DETECTORS.get(category, detect_structure_Regulations)
    blocks = detector(text)

    # Tables: collect and serialize
    tables_out = []
    for p in pages:
        for ti, t in enumerate(p.get("tables") or []):
            if not t:
                continue
            md_chunks = serialize_table(t)
            for ci, md in enumerate(md_chunks):
                tables_out.append({
                    "table_id": f"{document_id}/tbl_{len(tables_out)}",
                    "page": p.get("page", 0),
                    "table_index_on_page": ti,
                    "rows": len(t),
                    "cols": max((len(r) for r in t), default=0),
                    "markdown": md,
                    "in_block_id": None,
                })

    # References
    short_names = detect_short_names(text)
    refs = extract_references(text, short_names)
    refs = resolve_references([Reference(**r.__dict__) if hasattr(r, '__dict__') else r for r in refs], doc_index, document_id)
    # Convert Reference objects to dicts
    refs_dicts = [
        {
            "type": r.type,
            "raw_text": r.raw_text,
            "canonical": r.canonical,
            "target_document_id": r.target_document_id,
            "target_match_confidence": r.target_match_confidence,
            "occurrences": r.occurrences,
        }
        for r in refs
    ]

    # structured_text
    meta_for_xml = {
        "document_id": document_id,
        "document_type": document_type,
        "title": title,
        "issued_date": dates["issued_date"],
        "last_amended_date": dates["last_amended_date"],
        "domain": domain,
        "references": refs_dicts,
    }
    structured_text = build_structured_text(meta_for_xml, blocks, tables_out)

    # Segments (estimate total_segments for preamble; actual count == len(segments))
    # Two-pass: first pass to count, second to fill preambles with the real total.
    rough_segments = segment_blocks(blocks, MAX_TOKENS_DEFAULT, meta_for_xml, total_segments_estimate=0)
    total_segs = len(rough_segments)
    segments = segment_blocks(blocks, MAX_TOKENS_DEFAULT, meta_for_xml, total_segments_estimate=total_segs)

    # Structure summary
    block_types = Counter(b.name for b in blocks)
    structure_summary = {
        "total_blocks": len(blocks),
        "block_types": dict(block_types),
        "max_depth": 1,  # flat for now
    }

    result = {
        "document_id": document_id,
        "document_type": document_type,
        "source": {
            "category": category,
            "source_file": source_file,
            "source_urls": src.get("source_urls", {}),
            "input_json_path": str(input_path.resolve().relative_to(SOURCE_ROOT)),
        },
        "title": normalize_whitespace(title),
        "year": year,
        "issued_date": dates["issued_date"],
        "last_amended_date": dates["last_amended_date"],
        "status": "ok",
        "domain": domain,
        "references": refs_dicts,
        "short_names_defined": short_names,
        "text": text,
        "structured_text": structured_text,
        "segments": segments,
        "tables": tables_out,
        "structure_summary": structure_summary,
        "cleaning": cleaning,
    }

    # Validation
    result["validation"] = validate_document(result, blocks, text)

    return result


def build_doc_index(all_input_paths: list) -> dict:
    """First pass: build {document_id: {title, year, category, domain}} for cross-ref resolution."""
    index = {}
    for p in all_input_paths:
        try:
            src = json.loads(p.read_text())
        except Exception:
            continue
        category = src.get("category", "")
        year = src.get("year", "")
        source_file = src.get("source_file", "")
        title = src.get("title", p.stem)
        document_id = build_document_id(category, year, source_file)
        domain = classify_domain(title, (src.get("pages") or [{}])[0].get("text", "")[:2000] if src.get("pages") else "")
        index[document_id] = {"title": title, "year": year, "category": category, "domain": domain}
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated category names (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cats = sorted(d.name for d in SOURCE_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cats = [c for c in cats if c in wanted]

    print(f"Canonicalizing {SOURCE_ROOT} -> {OUTPUT_ROOT}")
    print(f"Categories: {cats}")

    # Collect all input files
    all_paths = []
    for cat in cats:
        cat_dir = SOURCE_ROOT / cat
        for p in cat_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            all_paths.append((cat, p))

    print(f"Total documents: {len(all_paths)}")

    # First pass: build doc index for cross-reference resolution
    print("Building document index for cross-reference resolution...")
    doc_index = build_doc_index([p for _, p in all_paths])
    print(f"  indexed {len(doc_index)} documents")

    # Second pass: process each doc
    by_cat = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, p, doc_index): (cat, p) for cat, p in all_paths}
        done = 0
        for fut in as_completed(futures):
            cat, p = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  FAILED: {cat}/{p.name}: {e}")
                continue
            out_dir = OUTPUT_ROOT / cat
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / p.name
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            by_cat[cat].append(result)
            done += 1
            if done % 20 == 0 or done == len(all_paths):
                print(f"  {done}/{len(all_paths)} done")

    # Write per-category indexes
    for cat, docs in by_cat.items():
        index = {
            "category": cat,
            "total_documents": len(docs),
            "documents": sorted([
                {
                    "document_id": d["document_id"],
                    "title": d["title"],
                    "year": d["year"],
                    "domain": d["domain"],
                    "issued_date": d.get("issued_date"),
                    "last_amended_date": d.get("last_amended_date"),
                    "tokens_estimate": estimate_tokens(d.get("text", "")),
                    "segment_count": len(d.get("segments", [])),
                    "reference_count": len(d.get("references", [])),
                    "output_path": f"{cat}/{Path(d['source']['source_file']).stem}.json" if d.get("source", {}).get("source_file") else "",
                }
                for d in docs
            ], key=lambda x: (x.get("year") or "9999", x.get("title") or "")),
        }
        (OUTPUT_ROOT / cat / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))

    # Write global report
    all_docs = [d for docs in by_cat.values() for d in docs]
    all_segs = [s for d in all_docs for s in d.get("segments", [])]
    all_refs = [r for d in all_docs for r in d.get("references", [])]
    all_seg_tokens = [s["tokens_estimate"] for s in all_segs]

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "input_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "corpus_stats": {
            "total_documents": len(all_docs),
            "total_tokens_estimate": sum(estimate_tokens(d.get("text", "")) for d in all_docs),
            "total_segments": len(all_segs),
            "total_references": len(all_refs),
            "total_tables": sum(len(d.get("tables", [])) for d in all_docs),
            "by_category": {
                cat: {
                    "docs": len(docs),
                    "tokens": sum(estimate_tokens(d.get("text", "")) for d in docs),
                    "segments": sum(len(d.get("segments", [])) for d in docs),
                    "references": sum(len(d.get("references", [])) for d in docs),
                }
                for cat, docs in by_cat.items()
            },
        },
        "validation_summary": {
            "avg_empty_tag_rate": round(statistics.mean(d["validation"]["empty_tag_rate"] for d in all_docs if d.get("validation")), 4),
            "avg_structural_coverage": round(statistics.mean(d["validation"]["structural_coverage"] for d in all_docs if d.get("validation")), 4),
            "docs_with_warnings": sum(1 for d in all_docs if d.get("validation", {}).get("warnings")),
            "segment_token_distribution": {
                "min": min(all_seg_tokens) if all_seg_tokens else 0,
                "p50": int(statistics.median(all_seg_tokens)) if all_seg_tokens else 0,
                "p95": int(statistics.quantiles(all_seg_tokens, n=20)[18]) if len(all_seg_tokens) > 1 else 0,
                "max": max(all_seg_tokens) if all_seg_tokens else 0,
            },
            "oversized_segments_total": sum(d["validation"]["oversized_segments"] for d in all_docs if d.get("validation")),
            "reference_resolution_rate": round(
                sum(1 for r in all_refs if r.get("target_document_id")) / max(1, len(all_refs)),
                4,
            ),
        },
        "edge_cases_handled": [
            {"document_id": d["document_id"], "issue": "empty_document"}
            for d in all_docs if d.get("status") == "empty"
        ],
    }
    (OUTPUT_ROOT / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Print summary
    print("\n========== PIPELINE SUMMARY ==========")
    print(f"  Documents: {len(all_docs)}")
    print(f"  Segments:  {len(all_segs)}")
    print(f"  References: {len(all_refs)}")
    print(f"  Tables:    {sum(len(d.get('tables', [])) for d in all_docs)}")
    print(f"  Avg structural coverage: {report['validation_summary']['avg_structural_coverage']:.1%}")
    print(f"  Avg empty tag rate: {report['validation_summary']['avg_empty_tag_rate']:.1%}")
    print(f"  Reference resolution rate: {report['validation_summary']['reference_resolution_rate']:.1%}")
    print(f"  Docs with warnings: {report['validation_summary']['docs_with_warnings']}")
    print(f"\nReport: {OUTPUT_ROOT / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
