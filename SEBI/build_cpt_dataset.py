#!/usr/bin/env python3
"""
Build CPT training dataset from SEBI_Canonical/.

Output: SEBI_CPT/train.jsonl, SEBI_CPT/valid.jsonl
Each line: {"text": "<structured_text content>"}

Splits 90/10 by document (not by segment) to avoid leakage.
"""
import json
from pathlib import Path
from random import Random

ROOT = Path(__file__).parent / "SEBI_Canonical"
OUT = Path(__file__).parent / "SEBI_CPT"
OUT.mkdir(parents=True, exist_ok=True)

# Collect all docs (skip indexes, skip empty docs)
docs = []
for p in sorted(ROOT.rglob("*.json")):
    if p.name.startswith("_") or p.name == "pipeline_report.json":
        continue
    d = json.loads(p.read_text())
    if d.get("status") != "ok" or not d.get("structured_text"):
        continue
    docs.append(d)

print(f"Loaded {len(docs)} docs with structured_text")

# Shuffle with fixed seed for reproducibility
rng = Random(42)
rng.shuffle(docs)

# 90/10 split by document
split = int(len(docs) * 0.9)
train_docs = docs[:split]
valid_docs = docs[split:]

print(f"Train: {len(train_docs)} docs, Valid: {len(valid_docs)} docs")

# Write JSONL
def write_jsonl(path, docs):
    total_chars = 0
    with open(path, "w") as f:
        for d in docs:
            text = d["structured_text"]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_chars += len(text)
    print(f"  {path.name}: {len(docs)} examples, {total_chars:,} chars")

write_jsonl(OUT / "train.jsonl", train_docs)
write_jsonl(OUT / "valid.jsonl", valid_docs)

# Also dump a small eval prompt set for manual checking
eval_prompts = [
    "What are the disclosure requirements for listed companies under SEBI LODR Regulations?",
    "Explain the regulatory framework for mutual funds in India.",
    "What penalties can SEBI impose for insider trading?",
    "Describe the registration process for stock brokers under SEBI regulations.",
    "What is the procedure for buy-back of securities under SEBI regulations?",
]
with open(OUT / "eval_prompts.jsonl", "w") as f:
    for p in eval_prompts:
        f.write(json.dumps({"prompt": p}) + "\n")
print(f"  eval_prompts.jsonl: {len(eval_prompts)} prompts")
