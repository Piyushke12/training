#!/usr/bin/env python3
"""
SEBI Legal Documents Scraper

Fetches legal documents (Acts, Rules, Regulations, General Orders,
Guidelines, Master Circulars) from sebi.gov.in and saves them in a
structured folder layout with per-category manifests.

Architecture:
  1. Listing page (HTML)        -> lists documents as <a> links to detail pages
  2. Listing pagination          -> POST to /sebiweb/ajax/home/getnewslistinfo.jsp
  3. Detail page (HTML)         -> contains an <iframe src=...> with the PDF URL
  4. PDF download               -> direct GET on the iframe src
"""

import os
import re
import sys
import json
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# SEBI does not publish a rate-limit policy and does not send Retry-After
# headers, but government sites commonly IP-block aggressive crawlers. We:
#   - Cap concurrency at a conservative 3 workers
#   - Enforce a per-thread min delay between requests (RATE_LIMIT_MIN_GAP_S)
#   - Add small jitter so requests don't fall into a lockstep rhythm
#   - Honor 429 / 503 with exponential backoff (see RateLimitAdapter below)
RATE_LIMIT_MIN_GAP_S = 1.5      # at least 1.5s between requests per thread
RATE_LIMIT_JITTER_S = 0.75      # +0..0.75s random jitter on top
RATE_LIMIT_429_BACKOFF_S = 30   # initial backoff on 429, doubles each retry

# Token bucket shared across threads: caps the global request rate regardless
# of worker count. Refills smoothly so we never exceed MAX_RATE_PER_SEC globally.
MAX_RATE_PER_SEC = 1.5  # ~90 requests/minute site-wide cap


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float = 4.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait, 2.0))


_bucket = TokenBucket(MAX_RATE_PER_SEC)


class RateLimitAdapter(HTTPAdapter):
    """HTTPAdapter that backs off on 429/503 with Retry-After awareness."""

    def send(self, request, **kwargs):
        backoff = RATE_LIMIT_429_BACKOFF_S
        for attempt in range(6):
            resp = super().send(request, **kwargs)
            if resp.status_code in (429, 503):
                ra = resp.headers.get("Retry-After")
                delay = float(ra) if ra and ra.isdigit() else backoff
                time.sleep(delay + random.uniform(0, 1.0))
                backoff *= 2
                continue
            return resp
        return resp

BASE = "https://www.sebi.gov.in"
LISTING_URL = f"{BASE}/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid={{ssid}}&smid=0"
AJAX_URL = f"{BASE}/sebiweb/ajax/home/getnewslistinfo.jsp"

# ssid -> (folder_name, page_size)
# Page size is 25 for every category; only Master Circulars and Circulars paginate.
# Master Circulars has 135 records / 25 = 6 pages.
# Circulars is excluded per user choice.
CATEGORIES = [
    {"ssid": 1, "name": "Acts",             "label": "Acts"},
    {"ssid": 2, "name": "Rules",             "label": "Rules"},
    {"ssid": 3, "name": "Regulations",       "label": "Regulations"},
    {"ssid": 4, "name": "General Orders",    "label": "General Orders"},
    {"ssid": 5, "name": "Guidelines",        "label": "Guidelines"},
    {"ssid": 6, "name": "Master Circulars",  "label": "Master Circulars"},
]

OUTPUT_ROOT = Path(__file__).parent / "SEBI_Documents"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# Module-level session per thread (requests.Session is not thread-safe across threads,
# but each worker gets its own via thread-local storage).
_tls = threading.local()


def get_session() -> requests.Session:
    """Return a thread-local requests.Session with retries configured."""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=(500, 502, 504),  # 429/503 handled by RateLimitAdapter
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
        )
        adapter = RateLimitAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{BASE}/legal.html",
        })
        _tls.session = s
        _tls.last_request_at = 0.0
    return _tls.session


def throttled_request(method: str, url: str, **kwargs):
    """Wrap a request with global token-bucket + per-thread min-gap pacing."""
    _bucket.acquire()
    session = get_session()
    # Per-thread minimum gap + jitter.
    last = getattr(_tls, "last_request_at", 0.0)
    elapsed = time.monotonic() - last
    wait = RATE_LIMIT_MIN_GAP_S - elapsed
    if wait > 0:
        time.sleep(wait + random.uniform(0, RATE_LIMIT_JITTER_S))
    _tls.last_request_at = time.monotonic()
    return session.request(method, url, **kwargs)


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_REPEATED_DOTS = re.compile(r'\.{2,}')
_REPEATED_SPACES = re.compile(r'\s+')
# Stray opening/closing square or curly brackets that often come from
# truncated titles like "...amended on March 13, 2026]".
_UNMATCHED_BRACKETS = re.compile(r'[][]|{}')


def sanitize_filename(name: str, max_len: int = 180) -> str:
    """Make a filesystem-safe, human-readable filename (no extension)."""
    name = name.strip()
    # Strip characters that are illegal on Windows/macOS/Linux filesystems.
    name = _INVALID_CHARS.sub("_", name)
    # Drop unmatched brackets from truncated titles.
    name = _UNMATCHED_BRACKETS.sub("", name)
    name = _REPEATED_DOTS.sub(".", name)
    name = _REPEATED_SPACES.sub(" ", name)
    name = name.strip(" .-_")
    if len(name) > max_len:
        name = name[:max_len].rsplit(" ", 1)[0].strip(" .-_")
    return name or "untitled"


# ---------------------------------------------------------------------------
# Listing-page parsing
# ---------------------------------------------------------------------------

# Match document detail links in the listing tables. Attribute quotes can be
# either single or double — the AJAX paginated responses use single quotes
# while the initial listing page uses double quotes.
# Examples:
#   <a href="https://www.sebi.gov.in/legal/acts/aug-2015/..._30609.html" target="_blank">Title</a>
#   <a href='https://www.sebi.gov.in/legal/master-circulars/sep-2024/..._86929.html'  target="_blank" title="..." class='points'>Title</a>
DOC_LINK_RE = re.compile(
    r'''<a\s+href=["'](https?://www\.sebi\.gov\.in/legal/[^"']+\.html)["']\s+target=["']_blank["'][^>]*>([^<]+)</a>''',
    re.IGNORECASE,
)

# Row years come in two formats:
#   <td>2015</td>                       (Acts/Rules/Regulations)
#   <td>Sep 23, 2024</td>               (Master Circulars/Circulars)
# Pull the 4-digit year out of either.
YEAR_RE = re.compile(r"<td[^>]*>\s*.*?\b(\d{4})\b.*?\s*</td>", re.IGNORECASE | re.DOTALL)


def parse_listing(html: str) -> list[dict]:
    """Return [{year, title, detail_url}, ...] from a listing HTML page."""
    docs: list[dict] = []
    # Walk row-by-row so we can attribute the year from the preceding <td>.
    rows = re.split(r"<tr[^>]*>", html, flags=re.IGNORECASE)
    for row in rows:
        links = DOC_LINK_RE.findall(row)
        if not links:
            continue
        year_match = YEAR_RE.search(row)
        year = year_match.group(1) if year_match else ""
        for url, title in links:
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue
            docs.append({"year": year, "title": title, "detail_url": url})
    return docs


def fetch_listing(ssid: int, label: str) -> list[dict]:
    """Fetch the listing page (with pagination) and return all documents."""
    first_url = LISTING_URL.format(ssid=ssid)
    r = throttled_request("GET", first_url, timeout=30)
    r.raise_for_status()
    html = r.text
    docs = parse_listing(html)

    # Detect total records to know how many pages we need.
    m = re.search(r"of\s+(\d+)\s+records", html)
    total = int(m.group(1)) if m else len(docs)
    print(f"  [{label}] listing: {len(docs)} on page 1, total reported = {total}")

    # Paginate via the AJAX endpoint.
    # Each page returns 25 records. The server's pagination uses nextValue as
    # a 1-indexed PAGE NUMBER (not a record offset) and doDirect as page-1.
    # We confirmed this empirically: nextValue=3, doDirect=2 -> records 51-75.
    page = 2
    while len(docs) < total:
        payload = {
            "nextValue": str(page),
            "next": "n",
            "search": "",
            "fromDate": "",
            "toDate": "",
            "fromYear": "",
            "toYear": "",
            "deptId": "",
            "sid": "1",
            "ssid": str(ssid),
            "smid": "0",
            "ssidhidden": str(ssid),
            "intmid": "-1",
            "sText": "Legal",
            "ssText": label,
            "smText": "",
            "doDirect": str(page - 1),
        }
        try:
            r = throttled_request("POST", AJAX_URL, data=payload, timeout=30,
                                   headers={"Referer": first_url, "X-Requested-With": "XMLHttpRequest"})
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    page {page} fetch failed: {e}")
            break
        page_docs = parse_listing(r.text)
        if not page_docs:
            print(f"    page {page}: no documents parsed, stopping")
            break
        docs.extend(page_docs)
        print(f"    page {page}: +{len(page_docs)} (total collected {len(docs)})")
        page += 1
        if page > 200:  # safety cap
            break

    # De-duplicate by detail_url (a document can appear twice if pagination overlaps).
    seen = set()
    unique = []
    for d in docs:
        if d["detail_url"] in seen:
            continue
        seen.add(d["detail_url"])
        unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Detail-page parsing
# ---------------------------------------------------------------------------

IFRAME_SRC_RE = re.compile(
    r'<iframe[^>]+src=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def extract_pdf_url(detail_html: str, detail_url: str) -> str | None:
    """Pull the PDF URL out of the iframe src on a detail page."""
    m = IFRAME_SRC_RE.search(detail_html)
    if not m:
        return None
    src = m.group(1)
    # The src is sometimes a viewer URL like /web/?file=<PDF_URL>
    if "file=" in src:
        qs = parse_qs(urlparse(src).query)
        if "file" in qs:
            return urljoin(BASE, qs["file"][0])
    # Otherwise it is a relative or absolute path to the PDF.
    return urljoin(detail_url, src)


def fetch_detail_and_pdf(doc: dict, cat_folder: Path) -> dict:
    """Fetch the detail page, extract the PDF URL, and download the PDF.

    Returns the doc dict enriched with pdf_url, local_filename, status.
    """
    detail_url = doc["detail_url"]
    result = {**doc, "pdf_url": None, "local_filename": None, "status": "pending"}

    try:
        r = throttled_request("GET", detail_url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        result["status"] = f"detail_fetch_failed: {e}"
        return result

    pdf_url = extract_pdf_url(r.text, detail_url)
    if not pdf_url:
        # Some older SEBI documents are published as inline HTML, not PDFs.
        # Save the detail page itself as .html so we keep the content.
        title_part = sanitize_filename(doc["title"])
        if doc.get("year"):
            html_name = f"{doc['year']} - {title_part}"
        else:
            html_name = title_part
        html_name = sanitize_filename(html_name, max_len=220) + ".html"
        out_path = cat_folder / html_name
        try:
            out_path.write_bytes(r.content)
        except OSError as e:
            result["status"] = f"html_save_failed: {e}"
            return result
        result["local_filename"] = html_name
        result["status"] = "saved_as_html"
        return result
    result["pdf_url"] = pdf_url

    # Build a safe filename: <sanitized title>.pdf
    # Include the year prefix when we have it for sorting.
    title_part = sanitize_filename(doc["title"])
    if doc.get("year"):
        filename = f"{doc['year']} - {title_part}"
    else:
        filename = title_part
    filename = sanitize_filename(filename, max_len=220) + ".pdf"
    # Avoid collisions inside a category by appending a short hash of the URL.
    suffix = abs(hash(detail_url)) % 100000
    out_path = cat_folder / filename
    if out_path.exists():
        # If the file already exists with the right size, skip re-downloading.
        try:
            hr = throttled_request("HEAD", pdf_url, timeout=20)
            remote_size = int(hr.headers.get("Content-Length", "0"))
        except requests.RequestException:
            remote_size = 0
        if remote_size and out_path.stat().st_size == remote_size:
            result["local_filename"] = filename
            result["status"] = "already_present"
            return result
        # Name collision with different content -> suffix it.
        stem, ext = os.path.splitext(filename)
        out_path = cat_folder / f"{stem}_{suffix}{ext}"

    try:
        with throttled_request("GET", pdf_url, timeout=60, stream=True) as pr:
            pr.raise_for_status()
            ct = pr.headers.get("Content-Type", "")
            if "pdf" not in ct.lower() and not pdf_url.lower().endswith(".pdf"):
                result["status"] = f"not_a_pdf: {ct}"
                return result
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in pr.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.rename(out_path)
    except requests.RequestException as e:
        result["status"] = f"pdf_download_failed: {e}"
        return result

    result["local_filename"] = out_path.name
    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_category(cat: dict, workers: int) -> dict:
    name = cat["name"]
    ssid = cat["ssid"]
    label = cat["label"]
    cat_folder = OUTPUT_ROOT / name
    cat_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} (ssid={ssid}) ===")

    docs = fetch_listing(ssid, label)
    print(f"  collected {len(docs)} unique documents for {name}")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_detail_and_pdf, d, cat_folder): d for d in docs}
        done = 0
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = {**futures[fut], "status": f"exception: {e}",
                       "pdf_url": None, "local_filename": None}
            results.append(res)
            done += 1
            if done % 25 == 0 or done == len(docs):
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"    progress {done}/{len(docs)}  (ok={ok})")

    manifest_path = cat_folder / "manifest.json"
    manifest = {
        "category": name,
        "ssid": ssid,
        "source_listing_url": LISTING_URL.format(ssid=ssid),
        "total_documents": len(results),
        "documents": sorted(results, key=lambda r: (r.get("year") or "9999", r.get("title") or "")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  manifest written: {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel download workers (default 3, kept low to be gentle on SEBI)")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated category names to run (default: all)")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    wanted = {c["name"] for c in CATEGORIES}
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    cats = [c for c in CATEGORIES if c["name"] in wanted]
    if not cats:
        print("No matching categories. Available:", [c["name"] for c in CATEGORIES])
        sys.exit(1)

    overall = []
    for cat in cats:
        overall.append(run_category(cat, args.workers))

    print("\n========== SUMMARY ==========")
    total_docs = 0
    total_ok = 0
    for m in overall:
        ok = sum(1 for d in m["documents"] if d["status"] == "ok")
        present = sum(1 for d in m["documents"] if d["status"] == "already_present")
        as_html = sum(1 for d in m["documents"] if d["status"] == "saved_as_html")
        failed = m["total_documents"] - ok - present - as_html
        total_docs += m["total_documents"]
        total_ok += ok + present + as_html
        print(f"  {m['category']:<20} {m['total_documents']:>4} docs  ok={ok}  cached={present}  html={as_html}  failed={failed}")
    print(f"  {'TOTAL':<20} {total_docs:>4} docs  succeeded={total_ok}")


if __name__ == "__main__":
    main()
