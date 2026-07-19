#!/usr/bin/env python3
"""
SEBI PDF -> JSON Extractor

Walks every PDF (and HTML) under SEBI_Documents/, extracts text and tables
page-by-page, and writes a structured JSON file per document into an
parallel SEBI_Extracted/ tree that mirrors the source layout.

Output JSON schema per document:
{
  "category": "Acts",
  "year": "1992",
  "title": "Securities and Exchange Board of India Act, 1992 ...",
  "source_pdf": "Acts/1992 - Securities and Exchange ....pdf",
  "source_urls": {"detail_url": "...", "pdf_url": "..."},
  "total_pages": 47,
  "total_chars": 123456,
  "total_tables": 8,
  "pages": [
    {"page": 1, "text": "...", "tables": [[[...],[...]], ...]},
    ...
  ]
}
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
import fitz  # PyMuPDF — faster and handles more PDF variants than pdfminer
import pytesseract
from PIL import Image
import io
from bs4 import BeautifulSoup

SOURCE_ROOT = Path(__file__).parent / "SEBI_Documents"
OUTPUT_ROOT = Path(__file__).parent / "SEBI_Extracted"


def load_manifest(cat_folder: Path) -> dict:
    """Read the manifest.json for a category, or return empty if missing."""
    mf = cat_folder / "manifest.json"
    if not mf.exists():
        return {"documents": []}
    return json.loads(mf.read_text())


def doc_metadata(pdf_path: Path, manifest: dict) -> dict:
    """Look up the manifest entry for a given PDF by matching local_filename."""
    name = pdf_path.name
    for d in manifest.get("documents", []):
        if d.get("local_filename") == name:
            return d
    return {}


def extract_pdf(pdf_path: Path) -> dict:
    """Extract text + tables from a PDF.

    Uses PyMuPDF (fitz) for text extraction — it handles PDFs that pdfplumber's
    underlying pdfminer.six reports as 0-page (cross-reference streams, object
    streams). Tables still come from pdfplumber, which has better table heuristics.
    Pages with images but no text layer are OCR'd via tesseract as a fallback.
    """
    pages_out = []
    total_chars = 0
    total_tables = 0
    ocr_pages = []

    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    page_texts = []
    page_has_images = []
    for i in range(page_count):
        page = doc[i]
        txt = page.get_text() or ""
        page_texts.append(txt)
        page_has_images.append(len(page.get_images()) > 0)
    doc.close()

    # Tables via pdfplumber (skip if it can't open the file — text is still captured).
    page_tables = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                tables = [t for t in tables if t and any(any(c for c in row) for row in t)]
                if tables:
                    page_tables[i] = tables
                    total_tables += len(tables)
    except Exception:
        pass

    # OCR fallback: render pages that have images but no/empty text.
    doc = fitz.open(pdf_path)
    for i in range(page_count):
        if len(page_texts[i].strip()) < 10 and page_has_images[i]:
            try:
                pix = doc[i].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                if ocr_text.strip():
                    page_texts[i] = ocr_text
                    ocr_pages.append(i + 1)
            except Exception:
                pass  # leave text empty; page recorded as unextractable
    doc.close()

    for i in range(page_count):
        text = page_texts[i]
        total_chars += len(text)
        tables = page_tables.get(i + 1, [])
        pages_out.append({
            "page": i + 1,
            "text": text,
            "tables": tables,
        })

    return {
        "pages": pages_out,
        "total_pages": len(pages_out),
        "total_chars": total_chars,
        "total_tables": total_tables,
        "ocr_pages": ocr_pages,
    }


def extract_html(html_path: Path) -> dict:
    """Extract text from an HTML-only document (no PDF exists on SEBI)."""
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    # The SEBI detail page wraps the actual content in <div class='table-scrollable'>.
    content = soup.find("div", class_="table-scrollable") or soup
    # Convert to plain text, preserving paragraph breaks.
    for br in content.find_all("br"):
        br.replace_with("\n")
    for p in content.find_all(["p", "div", "li"]):
        p.append("\n")
    text = content.get_text(separator=" ")
    # Collapse runs of whitespace but keep newlines.
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return {
        "pages": [{"page": 1, "text": text, "tables": []}],
        "total_pages": 1,
        "total_chars": len(text),
        "total_tables": 0,
    }


def process_one(src_path: Path, cat: str, manifest: dict) -> dict:
    """Extract one document and write its JSON. Returns a status dict."""
    meta = doc_metadata(src_path, manifest)
    out_dir = OUTPUT_ROOT / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = src_path.stem + ".json"
    out_path = out_dir / out_name

    result = {
        "category": cat,
        "year": meta.get("year", ""),
        "title": meta.get("title", src_path.stem),
        "source_file": f"{cat}/{src_path.name}",
        "source_urls": {
            "detail_url": meta.get("detail_url", ""),
            "pdf_url": meta.get("pdf_url", ""),
        },
        "status": "ok",
    }

    try:
        if src_path.suffix.lower() == ".pdf":
            extracted = extract_pdf(src_path)
        else:
            extracted = extract_html(src_path)
    except Exception as e:
        result["status"] = f"failed: {e}"
        result["error"] = traceback.format_exc()
        return result

    if extracted.get("ocr_pages"):
        result["ocr_pages"] = extracted["ocr_pages"]
        result["note"] = "Some pages were scanned images and were OCR'd; OCR text may be less accurate."

    result.update(extracted)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_category(cat: str, workers: int) -> dict:
    cat_folder = SOURCE_ROOT / cat
    if not cat_folder.exists():
        print(f"  {cat}: source folder missing, skipping")
        return {"category": cat, "ok": 0, "failed": 0}
    manifest = load_manifest(cat_folder)
    files = [p for p in cat_folder.iterdir()
             if p.is_file() and p.suffix.lower() in (".pdf", ".html")]
    print(f"\n=== {cat} ({len(files)} files) ===")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, p, cat, manifest): p for p in files}
        done = 0
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = {"category": cat, "status": f"exception: {e}",
                       "source_file": futures[fut].name}
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(files):
                ok = sum(1 for r in results if r["status"] == "ok")
                chars = sum(r.get("total_chars", 0) for r in results if r["status"] == "ok")
                tbls = sum(r.get("total_tables", 0) for r in results if r["status"] == "ok")
                print(f"  {done}/{len(files)}  ok={ok}  chars={chars:,}  tables={tbls}")

    # Write a category-level index summarizing every extracted document.
    index_path = OUTPUT_ROOT / cat / "_index.json"
    index = {
        "category": cat,
        "total_documents": len(results),
        "documents": [
            {
                "year": r.get("year", ""),
                "title": r.get("title", ""),
                "source_file": r.get("source_file", ""),
                "json_file": r["source_file"].split("/", 1)[-1].rsplit(".", 1)[0] + ".json"
                              if r.get("source_file") else "",
                "total_pages": r.get("total_pages", 0),
                "total_chars": r.get("total_chars", 0),
                "total_tables": r.get("total_tables", 0),
                "status": r.get("status", ""),
            }
            for r in sorted(results, key=lambda x: (x.get("year") or "9999",
                                                    x.get("title") or ""))
        ],
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = [r for r in results if r["status"] != "ok"]
    return {"category": cat, "ok": ok, "failed": len(failed), "failures": failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel extraction workers (default 4)")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated category names (default: all)")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cats = [d.name for d in SOURCE_ROOT.iterdir() if d.is_dir()]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cats = [c for c in cats if c in wanted]
    print(f"Extracting from {SOURCE_ROOT}")
    print(f"Output to {OUTPUT_ROOT}")
    print(f"Categories: {cats}")

    overall = []
    for cat in cats:
        overall.append(run_category(cat, args.workers))

    print("\n========== EXTRACTION SUMMARY ==========")
    total_docs = total_ok = 0
    for r in overall:
        total_docs += r["ok"] + r["failed"]
        total_ok += r["ok"]
        fail_info = f"  failures: {[f['source_file'] for f in r.get('failures', [])][:3]}" if r["failed"] else ""
        print(f"  {r['category']:<20} ok={r['ok']:>3}  failed={r['failed']}{fail_info}")
    print(f"  {'TOTAL':<20} ok={total_ok}/{total_docs}")


if __name__ == "__main__":
    main()
