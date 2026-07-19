#!/usr/bin/env python3
"""Parallel ingestion: 4 worker processes, each loads its own BGE model and
embeds a slice of chunks. Qdrant collection is created up front by the main
process, then workers upsert concurrently (Qdrant handles concurrent upserts).

Usage: python3 ingest_qdrant_parallel.py
"""
import argparse
import json
import os
import sys
import time
import torch
import multiprocessing as mp
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION = "sebi_docs"
QDRANT_URL = "http://localhost:6333"
CHUNKS_FILE = Path(__file__).parent / "SEBI_CPT" / "chunks.jsonl"
BATCH = 64


def setup_collection():
    """Main process: create collection + payload indexes."""
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"  deleting existing {COLLECTION}")
        client.delete_collection(COLLECTION)
    print(f"Creating {COLLECTION}...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    client.create_payload_index(COLLECTION, "doc_type", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "year", models.PayloadSchemaType.INTEGER)
    client.create_payload_index(COLLECTION, "doc_id", models.PayloadSchemaType.KEYWORD)


def worker(worker_id: int, chunk_slice: list, start_id: int, progress_queue: mp.Queue):
    """Each worker: load its own model + Qdrant client, embed+upsert its slice."""
    # Limit torch threads per worker — 12 cores / 4 workers = 3 threads each
    torch.set_num_threads(3)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"[w{worker_id}] loading BGE...", flush=True)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    model.max_seq_length = 512
    client = QdrantClient(url=QDRANT_URL)
    print(f"[w{worker_id}] ready, processing {len(chunk_slice)} chunks starting id={start_id}", flush=True)

    t0 = time.time()
    done = 0
    for i in range(0, len(chunk_slice), BATCH):
        batch = chunk_slice[i:i + BATCH]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH, convert_to_numpy=True)
        points = []
        for j, c in enumerate(batch):
            points.append(models.PointStruct(
                id=start_id + i + j + 1,
                vector=vectors[j].tolist(),
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "doc_type": c["doc_type"],
                    "doc_title": c["doc_title"],
                    "year": c["year"],
                    "block_kind": c["block_kind"],
                    "block_title": c["block_title"],
                    "text": c["text"],
                    "char_count": c["char_count"],
                },
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        done += len(batch)
        elapsed = time.time() - t0
        progress_queue.put((worker_id, done, len(chunk_slice), elapsed))
    progress_queue.put((worker_id, -1, len(chunk_slice), time.time() - t0))  # done signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks, splitting across {args.workers} workers")

    setup_collection()

    # Split chunks across workers
    slice_size = (len(chunks) + args.workers - 1) // args.workers
    slices = []
    for w in range(args.workers):
        start = w * slice_size
        end = min(start + slice_size, len(chunks))
        slices.append((chunks[start:end], start + 1))  # +1 so point IDs start at 1

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    progress_queue = mp.Queue()
    procs = []
    for w in range(args.workers):
        slice_chunks, start_id = slices[w]
        p = mp.Process(target=worker, args=(w, slice_chunks, start_id, progress_queue))
        p.start()
        procs.append(p)

    # Track progress
    t0 = time.time()
    worker_done = {w: False for w in range(args.workers)}
    totals = {w: [0, 0] for w in range(args.workers)}  # done, total
    total_done = 0
    while not all(worker_done.values()):
        try:
            wid, done, total, elapsed = progress_queue.get(timeout=5)
        except Exception:
            # Check if any worker died
            for w, p in enumerate(procs):
                if not p.is_alive() and not worker_done[w]:
                    print(f"[w{w}] DIED (exitcode={p.exitcode})", flush=True)
                    worker_done[w] = True
            continue

        if done == -1:
            worker_done[wid] = True
            print(f"[w{wid}] DONE in {elapsed:.1f}s", flush=True)
        else:
            totals[wid] = [done, total]
            total_done = sum(t[0] for t in totals.values())
            elapsed = time.time() - t0
            rate = total_done / max(elapsed, 0.1)
            eta = (len(chunks) - total_done) / max(rate, 0.1)
            # Only print from w0 to avoid log spam (aggregated)
            if wid == 0:
                per_worker = " ".join(f"{totals[w][0]}/{totals[w][1]}" for w in range(args.workers))
                print(f"  [{total_done}/{len(chunks)}] {rate:.1f} chunks/s, ETA {eta/60:.1f}m | {per_worker}", flush=True)

    for p in procs:
        p.join()

    print(f"\nAll workers done. Total: {time.time() - t0:.1f}s")
    client = QdrantClient(url=QDRANT_URL)
    count = client.count(COLLECTION, exact=True).count
    print(f"Points in {COLLECTION}: {count}")


if __name__ == "__main__":
    main()
