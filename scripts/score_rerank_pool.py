#!/usr/bin/env python3
"""Score each question with every chunk a reranker could be asked to reorder.

    python3 scripts/score_rerank_pool.py [--depth 30] [--device cuda] [options]

For every question of the gold set, the pool is the first DEPTH chunks of
BM25, of each dense encoder whose vectors are in data/interim/dense/
(scripts/encode_dense.py), and of each of their fusions, so any of them can
be reranked to that depth. The cross-encoder scores every pair of question
and chunk in the pool that has no score yet, and the scores go to
data/interim/rerank/bge-reranker-v2-m3.json. scripts/eval_hybrid.py then
reranks from those scores, without the model.

Needs the optional dependencies (pip install -e ".[dense]") and the model.
--model PATH loads it from a local folder instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.retrieve.bm25 import ChunkIndex
from daleel.retrieve.dense import (
    DENSE,
    ENCODERS,
    DenseIndex,
    VectorStore,
    device_name,
    model_revision,
    runtime_versions,
)
from daleel.retrieve.hybrid import Hybrid, fusions
from daleel.retrieve.rerank import (
    RERANK,
    RERANKER,
    RERANKER_NAME,
    ScoreStore,
    load_reranker,
    predict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the rerank pool with the cross-encoder.")
    parser.add_argument("--depth", type=int, default=30, help="chunks taken from each ranking")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--out", type=Path, default=RERANK)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: the best found)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default=None, metavar="PATH", help="load from a folder")
    args = parser.parse_args(argv)
    for path in (args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    chunks = read_chunks(args.chunks)
    texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    bm25 = ChunkIndex(chunks)
    dense = {}
    for name in sorted(ENCODERS):
        path = args.dense / f"{name}.npz"
        if path.exists():
            dense[name] = DenseIndex(chunks, VectorStore.load(path)).search
    if not dense:
        print(
            f"warning: no dense vectors in {args.dense}; the pool is BM25's alone", file=sys.stderr
        )
    retrievers = ["bm25", *dense]
    fused = fusions(bm25.search, dense)
    rankings = [Hybrid([search]) for search in [bm25.search, *dense.values()]]
    rankings += [Hybrid(searches) for searches in fused.values()]

    target = args.out / f"{RERANKER_NAME}.json"
    store = ScoreStore.load(target) if target.exists() else ScoreStore()
    pools = []
    for question in load_gold(args.gold):
        text = question["question"]
        pool = dict.fromkeys(
            chunk_id for ranking in rankings for chunk_id in ranking.search(text, args.depth)
        )
        missing = store.missing(text, [texts[chunk_id] for chunk_id in pool])
        if missing:
            pools.append((text, missing))
    pairs = sum(len(missing) for _, missing in pools)
    if not pairs:
        print(f"every pair is already scored in {target}")
        return 0

    started = time.perf_counter()
    model = load_reranker(args.device, args.model)
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    for text, missing in pools:
        store.add(text, missing, predict(model, text, missing, args.batch_size))
    seconds = time.perf_counter() - started
    store.meta = {
        "model": args.model or RERANKER,
        "revision": None if args.model else model_revision(RERANKER),
        "scores": "logits",
        "pool": {"depth": args.depth, "rankings": [*retrievers, *fused]},
        "device": device_name(model),
        "versions": runtime_versions(),
        "seconds_to_load": round(loaded, 2),
        "seconds_to_score": round(seconds, 2),
        "pairs_scored": pairs,
    }
    store.save(target)
    print(
        f"{pairs} pairs of {len(pools)} questions scored in {seconds:.1f} s on "
        f"{store.meta['device']}, pools from {', '.join(retrievers)} and their fusions, "
        f"written to {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
