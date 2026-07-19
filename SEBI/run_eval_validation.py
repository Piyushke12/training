#!/usr/bin/env python3
"""Run 5 validation questions on base vs adapted model, score against gold keywords."""
import json
import subprocess
import re
import sys
from pathlib import Path

PROMPTS_FILE = str(Path(__file__).parent / "SEBI_CPT" / "eval_validation.jsonl")
ADAPTER = str(Path(__file__).parent / "SEBI_CPT" / "adapters_exp")
MODEL = "Qwen/Qwen2.5-3B-Instruct"

def run_model(prompt, adapter=None):
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", MODEL,
        "--prompt", prompt,
        "--max-tokens", "400",
        "--temp", "0.7",
        "--seed", "42",
    ]
    if adapter:
        cmd.extend(["--adapter-path", adapter])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = r.stdout
    m = re.search(r"==========\n(.*?)\n==========", out, re.DOTALL)
    if m:
        return m.group(1).strip()
    return out

def score(response, gold_keywords):
    resp_lower = response.lower()
    hits = [kw for kw in gold_keywords if kw.lower() in resp_lower]
    return len(hits), len(gold_keywords), hits

def main():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    results = []
    for i, p in enumerate(prompts, 1):
        print(f"\n=== Q{i}: {p['prompt'][:80]}... ===", flush=True)
        print("  Running base...", flush=True)
        base_resp = run_model(p['prompt'])
        print("  Running adapted...", flush=True)
        adapt_resp = run_model(p['prompt'], adapter=ADAPTER)

        base_score = score(base_resp, p['gold_answer_keywords'])
        adapt_score = score(adapt_resp, p['gold_answer_keywords'])

        results.append({
            "q_num": i,
            "prompt": p['prompt'],
            "gold": p['gold_answer_keywords'],
            "base_response": base_resp,
            "adapted_response": adapt_resp,
            "base_score": base_score,
            "adapted_score": adapt_score,
        })

    out_path = str(Path(__file__).parent / "SEBI_CPT" / "eval_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== SCORE SUMMARY ===")
    print(f"{'Q':<3} {'Base':<10} {'Adapted':<10} {'Delta':<8}")
    for r in results:
        b = r['base_score']
        a = r['adapted_score']
        print(f"{r['q_num']:<3} {b[0]}/{b[1]:<6} {a[0]}/{a[1]:<7} {a[0]-b[0]:+d}")

if __name__ == "__main__":
    main()
