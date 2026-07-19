#!/usr/bin/env python3
"""Chunk canonical SEBI JSON docs into RAG-sized pieces.

Uses the XML structure in `structured_text` to find natural boundaries
(<regulation>, <schedule>, <chapter> tags), then splits any block longer
than MAX_CHARS into smaller pieces at paragraph boundaries.

Output: writes chunks to a JSONL file, one chunk per line.
"""
import json
import re
import sys
from pathlib import Path

CANONICAL_ROOT = Path(__file__).parent / "SEBI_Canonical"
OUTPUT_FILE = Path(__file__).parent / "SEBI_CPT" / "chunks.jsonl"

MAX_CHARS = 1500
MIN_CHARS = 200
OVERLAP_CHARS = 150


def extract_blocks(structured_text: str, doc_id: str, doc_title: str, doc_type: str, year: str):
    """Yield (block_kind, block_title, text) tuples from the structured XML."""
    pattern = re.compile(
        r"<(?P<tag>regulation|schedule|chapter|part|rule|section)\b[^>]*title=\"(?P<title>[^\"]*)\"[^>]*>(?P<body>.*?)(?=<(?:regulation|schedule|chapter|part|rule|section)\b|</document>)",
        re.DOTALL,
    )
    for m in pattern.finditer(structured_text):
        body = m.group("body").strip()
        body = re.sub(r"<[^>]+>", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            yield (m.group("tag"), m.group("title"), body)


def split_long_block(text: str, max_chars: int, overlap: int):
    """Split a long string at paragraph/sentence boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    sentences = re.split(r"(?<=[.;])\s+", text)
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if overlap > 0 and chunks:
                cur = chunks[-1][-overlap:] + " " + s
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks


def chunk_doc(path: Path):
    with open(path) as f:
        d = json.load(f)
    doc_id = d.get("document_id", path.stem)
    doc_title = d.get("title", path.stem)
    doc_type = d.get("document_type", "unknown")
    year = d.get("year", "")
    structured = d.get("structured_text", "")
    if not structured:
        return

    chunk_idx = 0
    for kind, title, body in extract_blocks(structured, doc_id, doc_title, doc_type, year):
        if len(body) < MIN_CHARS:
            continue
        for piece in split_long_block(body, MAX_CHARS, OVERLAP_CHARS):
            if len(piece) < MIN_CHARS:
                continue
            yield {
                "chunk_id": f"{doc_id}_{chunk_idx:04d}",
                "doc_id": doc_id,
                "doc_type": doc_type,
                "doc_title": doc_title,
                "year": year,
                "block_kind": kind,
                "block_title": title[:200],
                "text": piece,
                "char_count": len(piece),
            }
            chunk_idx += 1


def main():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    json_files = sorted(CANONICAL_ROOT.rglob("*.json"))
    json_files = [p for p in json_files if p.name != "_index.json" and "pipeline_report" not in p.name]
    print(f"Found {len(json_files)} canonical JSON files")

    total_chunks = 0
    per_type = {}
    with open(OUTPUT_FILE, "w") as out:
        for i, path in enumerate(json_files, 1):
            try:
                for chunk in chunk_doc(path):
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1
                    per_type[chunk["doc_type"]] = per_type.get(chunk["doc_type"], 0) + 1
            except Exception as e:
                print(f"  ERR {path.name}: {e}", file=sys.stderr)
            if i % 20 == 0 or i == len(json_files):
                print(f"  [{i}/{len(json_files)}] {total_chunks} chunks so far")

    print(f"\nTotal chunks: {total_chunks}")
    print("Per doc_type:")
    for k, v in sorted(per_type.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")

    sizes = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            sizes.append(json.loads(line)["char_count"])
    print(f"\nChunk size: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
