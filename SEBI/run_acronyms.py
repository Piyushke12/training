#!/usr/bin/env python3
"""Run acronym eval on base and adapted models without RAG."""
import json
import re
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = str(Path(__file__).parent / "SEBI_CPT" / "adapters_exp")
PROMPTS_FILE = str(Path(__file__).parent / "SEBI_CPT" / "eval_acronyms.jsonl")


def run_model(prompt, adapter=None):
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", MODEL,
        "--prompt", prompt,
        "--max-tokens", "300",
        "--temp", "0.3",
        "--seed", "42",
    ]
    if adapter:
        cmd.extend(["--adapter-path", adapter])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    m = re.search(r"==========\n(.*?)\n==========", r.stdout, re.DOTALL)
    return m.group(1).strip() if m else r.stdout


def score(response, gold_keywords):
    resp_lower = response.lower()
    hits = [kw for kw in gold_keywords if kw.lower() in resp_lower]
    return len(hits), len(gold_keywords), hits


def main():
    with open(PROMPTS_FILE) as f:
        prompts = [json.loads(l) for l in f if l.strip()]

    results = []
    for i, p in enumerate(prompts, 1):
        acronym = p["prompt"].split(" does ")[1].split(" ")[0]
        print(f"\n=== A{i}: {acronym} ===", flush=True)
        print("  base...", flush=True)
        base = run_model(p["prompt"])
        print("  adapted...", flush=True)
        adapted = run_model(p["prompt"], adapter=ADAPTER)
        bs = score(base, p["gold_answer_keywords"])
        as_ = score(adapted, p["gold_answer_keywords"])
        results.append({
            "acronym": acronym,
            "prompt": p["prompt"],
            "gold": p["gold_answer_keywords"],
            "base_response": base,
            "adapted_response": adapted,
            "base_score": list(bs),
            "adapted_score": list(as_),
        })
        print(f"  base: {bs[0]}/{bs[1]}  adapted: {as_[0]}/{as_[1]}", flush=True)

    out = str(Path(__file__).parent / "SEBI_CPT" / "eval_acronyms_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print("\n=== SUMMARY ===")
    print(f"{'Acronym':<8} {'Base':<8} {'Adapted':<8}")
    for r in results:
        print(f"{r['acronym']:<8} {r['base_score'][0]}/{r['base_score'][1]:<5} {r['adapted_score'][0]}/{r['adapted_score'][1]}")


if __name__ == "__main__":
    main()
