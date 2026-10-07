#!/usr/bin/env python3
"""Time retrieval per question, stage by stage, as it would run live.

    python3 scripts/time_retrieval.py --dense NAME [--dense NAME] [--rerank] [options]

Loads the models, then for every question of the gold set times each stage
of one retrieval: BM25, encoding the question, the dense search, fusion, and
reranking the head of the fused ranking. The chunks' vectors are read from
data/interim/dense/, since a live system encodes its chunks once, ahead of
any question. The first question is run once untimed, to warm the GPU.
Prints the median and 95th percentile of each stage and of the whole, in
milliseconds, and the device and versions it ran on.

Needs the optional dependencies (pip install -e ".[dense]") and the models.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.retrieve.bm25 import ChunkIndex
from daleel.retrieve.dense import (
    DENSE,
    ENCODERS,
    QUESTION,
    DenseIndex,
    VectorStore,
    device_name,
    encode,
    load_model,
    runtime_versions,
)
from daleel.retrieve.fusion import rrf
from daleel.retrieve.hybrid import CANDIDATES
from daleel.retrieve.rerank import RERANK_DEPTH, load_reranker, predict, rerank


def percentile(values: list[float], share: float) -> float:
    """The value below which `share` of the values fall, by the nearest rank."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered)) - 1))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time retrieval stage by stage.")
    parser.add_argument("--dense", action="append", choices=sorted(ENCODERS), default=[])
    parser.add_argument("--rerank", action="store_true", help="rerank the head of the ranking")
    parser.add_argument("--rerank-depth", type=int, default=RERANK_DEPTH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--vectors", type=Path, default=DENSE)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: the best found)")
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=PATH", help="load from a folder"
    )
    args = parser.parse_args(argv)
    paths = dict(item.split("=", 1) for item in args.model)

    chunks = read_chunks(args.chunks)
    texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    bm25 = ChunkIndex(chunks)
    encoders = []
    for name in args.dense:
        path = args.vectors / f"{name}.npz"
        if not path.exists():
            print(f"error: no {path}; run scripts/encode_dense.py first", file=sys.stderr)
            return 2
        model = load_model(ENCODERS[name], args.device, paths.get(name))
        encoders.append((ENCODERS[name], model, DenseIndex(chunks, VectorStore.load(path))))
    reranker = load_reranker(args.device, paths.get("reranker")) if args.rerank else None
    questions = [question["question"] for question in load_gold(args.gold)]

    times: dict[str, list[float]] = {}

    def run(question: str) -> dict[str, float]:
        spent = {}
        started = time.perf_counter()
        rankings = [bm25.search(question, CANDIDATES)]
        spent["bm25"] = time.perf_counter() - started
        for encoder, model, index in encoders:
            started = time.perf_counter()
            vector = encode(model, encoder, [question], QUESTION)[0]
            spent[f"encode {encoder.name}"] = time.perf_counter() - started
            started = time.perf_counter()
            rankings.append(index.nearest(vector, CANDIDATES))
            spent[f"search {encoder.name}"] = time.perf_counter() - started
        started = time.perf_counter()
        ranked = rrf(rankings) if len(rankings) > 1 else rankings[0]
        spent["fuse"] = time.perf_counter() - started
        if reranker is not None:
            started = time.perf_counter()

            def score(text: str, candidates: list[str]) -> list[float]:
                return predict(reranker, text, candidates)

            rerank(question, ranked, texts, score, args.rerank_depth)
            spent["rerank"] = time.perf_counter() - started
        spent["total"] = sum(spent.values())
        return spent

    run(questions[0])
    for question in questions:
        for stage, seconds in run(question).items():
            times.setdefault(stage, []).append(seconds * 1000)

    print(f"| stage, {len(questions)} questions | p50 ms | p95 ms |")
    print("|---|---|---|")
    for stage, values in times.items():
        median, p95 = statistics.median(values), percentile(values, 0.95)
        print(f"| {stage} | {median:.1f} | {p95:.1f} |")
    where = device_name(encoders[0][1]) if encoders else device_name(reranker)
    print(f"\non {where}; {runtime_versions()}; chunks {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
