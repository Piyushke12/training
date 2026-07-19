#!/usr/bin/env python3
"""
SEBI Extracted -> Clean

Reads every per-document JSON under SEBI_Extracted/, applies cleaning
transforms, and writes a parallel SEBI_Clean/ tree.

Transforms (defaults, all configurable via flags):
  1. Drop pages flagged as OCR'd (low-quality tesseract fallback text)
  2. Drop empty pages (no text AND no tables)
  3. Strip broken table rows (rows where every cell is None/empty)
  4. Strip Devanagari runs from text (English-only training default)

Writes:
  SEBI_Clean/<category>/<doc>.json   — cleaned per-document JSON
  SEBI_Clean/<category>/_index.json   — per-category index
  SEBI_Clean/cleaning_report.json     — global report of what was dropped
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent / "SEBI_Extracted"
OUTPUT_ROOT = Path(__file__).parent / "SEBI_Clean"

# Devanagari block U+0900–U+097F, plus Vedic Extensions and Devanagari Extended
# that sometimes appear alongside in the Gazette PDFs.
_DEVANAGARI_RE = re.compile(r'[ऀ-ॿ꣠-ꣿ᳐-᳿]+')


def clean_table(table: list) -> list:
    """Drop rows where every cell is None, empty, or whitespace-only."""
    out = []
    for row in table:
        if not row:
            continue
        if all(c is None or (isinstance(c, str) and not c.strip()) for c in row):
            continue
        out.append(row)
    return out


def clean_text(text: str, strip_devanagari: bool) -> str:
    if not text:
        return text
    if strip_devanagari:
        text = _DEVANAGARI_RE.sub('', text)
        # Collapse the gaps left behind by Devanagari removal.
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def clean_doc(doc: dict, strip_devanagari: bool) -> tuple[dict, dict]:
    """Return (cleaned_doc, stats)."""
    stats = {
        "original_pages": doc.get("total_pages", 0),
        "original_chars": doc.get("total_chars", 0),
        "original_tables": doc.get("total_tables", 0),
        "dropped_ocr_pages": 0,
        "dropped_empty_pages": 0,
        "original_table_rows": 0,
        "dropped_table_rows": 0,
        "devanagari_stripped": strip_devanagari,
    }

    ocr_pages = set(doc.get("ocr_pages", []))

    cleaned_pages = []
    for p in doc.get("pages", []):
        page_num = p["page"]
        text = p.get("text", "") or ""
        tables = p.get("tables") or []

        # 1. Drop OCR'd pages (low-quality tesseract output)
        if page_num in ocr_pages:
            stats["dropped_ocr_pages"] += 1
            continue

        # 2. Clean tables first so empty-page check can see if any content remains
        cleaned_tables = []
        for t in tables:
            stats["original_table_rows"] += len(t)
            cleaned = clean_table(t)
            stats["dropped_table_rows"] += len(t) - len(cleaned)
            if cleaned:
                cleaned_tables.append(cleaned)

        # 3. Clean text (strip Devanagari if requested)
        cleaned_text = clean_text(text, strip_devanagari)

        # 4. Drop empty pages (no text AND no tables after cleaning)
        if not cleaned_text.strip() and not cleaned_tables:
            stats["dropped_empty_pages"] += 1
            continue

        cleaned_pages.append({
            "page": page_num,
            "text": cleaned_text,
            "tables": cleaned_tables,
        })

    total_chars = sum(len(p["text"]) for p in cleaned_pages)
    total_tables = sum(len(p["tables"]) for p in cleaned_pages)

    out = {k: v for k, v in doc.items() if k not in ("pages", "ocr_pages", "note")}
    out["pages"] = cleaned_pages
    out["total_pages"] = len(cleaned_pages)
    out["total_chars"] = total_chars
    out["total_tables"] = total_tables
    out["cleaning"] = stats
    stats["cleaned_pages"] = len(cleaned_pages)
    stats["cleaned_chars"] = total_chars
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-devanagari", action="store_true",
                    help="preserve Devanagari (Hindi) text instead of stripping it")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated category names (default: all)")
    args = ap.parse_args()

    strip_devanagari = not args.keep_devanagari
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cats = sorted(d.name for d in SOURCE_ROOT.iterdir() if d.is_dir())
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cats = [c for c in cats if c in wanted]
    print(f"Cleaning {SOURCE_ROOT} -> {OUTPUT_ROOT}")
    print(f"Categories: {cats}")
    print(f"Devanagari: {'KEEP' if args.keep_devanagari else 'STRIP'}")

    report = {
        "transforms": {
            "drop_ocr_pages": True,
            "drop_empty_pages": True,
            "strip_broken_table_rows": True,
            "strip_devanagari": strip_devanagari,
        },
        "categories": [],
        "totals": {
            "docs": 0,
            "original_pages": 0,
            "cleaned_pages": 0,
            "original_chars": 0,
            "cleaned_chars": 0,
            "dropped_ocr_pages": 0,
            "dropped_empty_pages": 0,
            "dropped_table_rows": 0,
        },
    }

    for cat in cats:
        src_cat = SOURCE_ROOT / cat
        out_cat = OUTPUT_ROOT / cat
        out_cat.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in src_cat.glob("*.json") if not p.name.startswith("_"))
        print(f"\n=== {cat} ({len(files)} docs) ===")

        cat_docs = []
        cat_stats = {"docs": 0, "original_pages": 0, "cleaned_pages": 0,
                     "original_chars": 0, "cleaned_chars": 0,
                     "dropped_ocr_pages": 0, "dropped_empty_pages": 0,
                     "dropped_table_rows": 0}

        for jf in files:
            doc = json.loads(jf.read_text())
            cleaned, stats = clean_doc(doc, strip_devanagari)
            out_path = out_cat / jf.name
            out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2))

            for k in cat_stats:
                if k == "docs":
                    cat_stats[k] += 1
                elif k in stats:
                    cat_stats[k] += stats[k]

            cat_docs.append({
                "year": cleaned.get("year", ""),
                "title": cleaned.get("title", ""),
                "source_file": cleaned.get("source_file", ""),
                "json_file": jf.name,
                "original_pages": stats["original_pages"],
                "cleaned_pages": stats["cleaned_pages"],
                "original_chars": stats["original_chars"],
                "cleaned_chars": stats["cleaned_chars"],
                "dropped_ocr_pages": stats["dropped_ocr_pages"],
                "dropped_empty_pages": stats["dropped_empty_pages"],
                "dropped_table_rows": stats["dropped_table_rows"],
            })

        # Per-category index
        index = {
            "category": cat,
            "total_documents": len(cat_docs),
            "documents": sorted(cat_docs, key=lambda x: (x.get("year") or "9999",
                                                          x.get("title") or "")),
        }
        (out_cat / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))

        print(f"  pages: {cat_stats['original_pages']:,} -> {cat_stats['cleaned_pages']:,}")
        print(f"  chars: {cat_stats['original_chars']:,} -> {cat_stats['cleaned_chars']:,}")
        print(f"  dropped: ocr={cat_stats['dropped_ocr_pages']}  "
              f"empty={cat_stats['dropped_empty_pages']}  "
              f"table_rows={cat_stats['dropped_table_rows']}")

        report["categories"].append({"category": cat, **cat_stats})
        for k in report["totals"]:
            if k == "docs":
                report["totals"][k] += cat_stats.get("docs", 0)
            elif k in cat_stats:
                report["totals"][k] += cat_stats[k]

    (OUTPUT_ROOT / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    t = report["totals"]
    print("\n========== CLEANING SUMMARY ==========")
    print(f"  docs:                  {t['docs']}")
    print(f"  pages: {t['original_pages']:,} -> {t['cleaned_pages']:,}  "
          f"(dropped {t['original_pages'] - t['cleaned_pages']:,})")
    print(f"  chars: {t['original_chars']:,} -> {t['cleaned_chars']:,}  "
          f"(dropped {t['original_chars'] - t['cleaned_chars']:,})")
    print(f"  dropped OCR pages:     {t['dropped_ocr_pages']}")
    print(f"  dropped empty pages:   {t['dropped_empty_pages']}")
    print(f"  dropped table rows:    {t['dropped_table_rows']}")
    print(f"\nReport: {OUTPUT_ROOT / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
