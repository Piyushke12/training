#!/usr/bin/env python3
"""SEBI RAG agent.

Searches Qdrant for relevant chunks, then uses an LLM (base or adapted Qwen2.5-3B
via mlx_lm.generate) to synthesize an answer.

Usage:
    python3 sebi_rag.py --question "..." [--adapter PATH] [--top-k 5] [--show-context]
    python3 sebi_rag.py --eval eval_validation.jsonl --adapter PATH
    python3 sebi_rag.py --eval eval_validation.jsonl  # base model, no adapter
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION = "sebi_docs"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DEFAULT = str(Path(__file__).parent / "SEBI_CPT" / "adapters_exp")

# Module-level singletons — initialized lazily so the script doesn't load BGE
# if the user just wants --help.
_qdrant = None
_embedder = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading BGE embedder...", file=sys.stderr)
        _embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    return _embedder


def search(query: str, top_k: int = 5, doc_type: str | None = None):
    """Search Qdrant. Returns list of {text, doc_title, score, ...}."""
    embedder = get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    flt = None
    if doc_type:
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
        )
    res = get_qdrant().query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    ).points
    return [
        {
            "text": p.payload["text"],
            "doc_title": p.payload["doc_title"],
            "doc_type": p.payload["doc_type"],
            "block_kind": p.payload["block_kind"],
            "block_title": p.payload["block_title"],
            "score": p.score,
        }
        for p in res
    ]


def build_prompt(question: str, contexts: list, history: list = None) -> str:
    """Chat-template prompt with system instruction + retrieved context + question."""
    ctx_text = "\n\n".join(
        f"[{i+1}] ({c['doc_title'][:80]} — {c['block_kind']} {c['block_title'][:60]})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    system = (
        "You are a legal research assistant for SEBI regulations. "
        "Answer the user's question using ONLY the context passages below. "
        "If the answer is not in the context, say you don't have enough information. "
        "Cite passages by their [N] number. Be precise about numbers, timelines, and definitions."
    )
    user = f"Context:\n{ctx_text}\n\nQuestion: {question}\n\nAnswer (cite [N] for sources):"
    return system, user


def generate(system: str, user: str, adapter: str | None = None, max_tokens: int = 500) -> str:
    """Call mlx_lm.generate via subprocess."""
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", LLM_MODEL,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.3",  # low temp for factual
        "--seed", "42",
        "--ignore-chat-template",
    ]
    if adapter:
        cmd.extend(["--adapter-path", adapter])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = r.stdout
    m = re.search(r"==========\n(.*?)\n==========", out, re.DOTALL)
    return m.group(1).strip() if m else out


def answer_question(question: str, adapter: str | None = None, top_k: int = 5, show_context: bool = False):
    print(f"\n>>> Question: {question}", file=sys.stderr)
    contexts = search(question, top_k=top_k)
    print(f">>> Retrieved {len(contexts)} chunks (top score: {contexts[0]['score']:.3f})", file=sys.stderr)
    if show_context:
        for i, c in enumerate(contexts):
            print(f"\n  [{i+1}] score={c['score']:.3f} | {c['doc_title'][:60]} | {c['block_kind']} {c['block_title'][:40]}", file=sys.stderr)
            print(f"      {c['text'][:200]}...", file=sys.stderr)
    system, user = build_prompt(question, contexts)
    answer = generate(system, user, adapter=adapter)
    return answer, contexts


def run_eval(eval_file: str, adapter: str | None = None, top_k: int = 5):
    out_path = str(Path(__file__).parent / "SEBI_CPT" / ("eval_rag_" + ("adapted.json" if adapter else "base.json")))
    results = []
    with open(eval_file) as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    for i, p in enumerate(prompts, 1):
        print(f"\n=== Q{i}/{len(prompts)} ===", file=sys.stderr)
        ans, ctxs = answer_question(p["prompt"], adapter=adapter, top_k=top_k)
        # score
        resp_lower = ans.lower()
        hits = [kw for kw in p["gold_answer_keywords"] if kw.lower() in resp_lower]
        results.append({
            "q_num": i,
            "prompt": p["prompt"],
            "gold": p["gold_answer_keywords"],
            "rag_answer": ans,
            "top_contexts": [{"doc": c["doc_title"][:80], "block": c["block_title"][:60], "score": c["score"]} for c in ctxs],
            "score": [len(hits), len(p["gold_answer_keywords"]), hits],
        })
        print(f"    Score: {len(hits)}/{len(p['gold_answer_keywords'])} hits={hits}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"Q{r['q_num']}: {r['score'][0]}/{r['score'][1]} | {r['score'][2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", "-q", help="single question")
    ap.add_argument("--eval", help="eval jsonl file")
    ap.add_argument("--adapter", default=None, help="adapter path; omit for base model")
    ap.add_argument("--no-adapter", action="store_true", help="explicitly use base model")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    adapter = args.adapter
    if args.no_adapter:
        adapter = None
    elif adapter is None and Path(ADAPTER_DEFAULT).exists() and not args.eval:
        # default to adapted for interactive queries
        adapter = ADAPTER_DEFAULT

    if args.question:
        ans, ctxs = answer_question(args.question, adapter=adapter, top_k=args.top_k, show_context=args.show_context)
        print(ans)
    elif args.eval:
        run_eval(args.eval, adapter=adapter, top_k=args.top_k)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
