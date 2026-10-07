#!/usr/bin/env python3
"""Time retrieval per question, for every row of table T4, as it would run live.

    python3 scripts/time_retrieval.py --dense NAME [--dense NAME] [--rerank] [options]

Loads the models, then for every question of the gold set times each stage
of retrieval: BM25, encoding the question with each encoder and searching its
vectors, fusing each set of rankings that scripts/eval_hybrid.py fuses, and
reranking the head of each fusion. A row's time for a question is the sum of
the stages it runs, one after another. The chunks' vectors are read from
data/interim/dense/, since a live system encodes its chunks once, ahead of
any question. The first question is run once untimed, to warm the GPU.

Prints the median and 95th percentile of each row and of each stage, in
milliseconds, and the device and versions it ran on. Needs the optional
dependencies (pip install -e ".[dense]") and the models.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
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
from daleel.retrieve.hybrid import CANDIDATES, fusions
from daleel.retrieve.rerank import RERANK_DEPTH, load_reranker, predict, rerank


def percentile(values: list[float], share: float) -> float:
    """The value below which `share` of the values fall, by the nearest rank."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered)) - 1))]


def table(title: str, times: dict[str, list[float]]) -> None:
    print(f"| {title} | p50 ms | p95 ms |")
    print("|---|---|---|")
    for name, values in times.items():
        median, p95 = statistics.median(values), percentile(values, 0.95)
        print(f"| {name} | {median:.1f} | {p95:.1f} |")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time retrieval stage by stage.")
    parser.add_argument("--dense", action="append", choices=sorted(ENCODERS), default=[])
    parser.add_argument("--rerank", action="store_true", help="rerank the head of each fusion")
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
    for name in dict.fromkeys(args.dense):
        path = args.vectors / f"{name}.npz"
        if not path.exists():
            print(f"error: no {path}; run scripts/encode_dense.py first", file=sys.stderr)
            return 2
        model = load_model(ENCODERS[name], args.device, paths.get(name))
        encoders.append((ENCODERS[name], model, DenseIndex(chunks, VectorStore.load(path))))
    reranker = load_reranker(args.device, paths.get("reranker")) if args.rerank else None
    fused = fusions("BM25", {encoder.name: encoder.name for encoder, _, _ in encoders})
    questions = [question["question"] for question in load_gold(args.gold)]

    def score(question: str, candidates: Sequence[str]) -> list[float]:
        return predict(reranker, question, candidates)

    def run(question: str) -> tuple[dict[str, float], dict[str, float]]:
        """Seconds per stage, and per row of table T4, for one question."""
        stages, rankings = {}, {}
        started = time.perf_counter()
        rankings["BM25"] = bm25.search(question, CANDIDATES)
        stages["bm25"] = time.perf_counter() - started
        rows = {"BM25": stages["bm25"]}
        for encoder, model, index in encoders:
            started = time.perf_counter()
            vector = encode(model, encoder, [question], QUESTION)[0]
            stages[f"encode {encoder.name}"] = time.perf_counter() - started
            started = time.perf_counter()
            rankings[encoder.name] = index.nearest(vector, CANDIDATES)
            stages[f"search {encoder.name}"] = time.perf_counter() - started
            rows[encoder.name] = sum(
                stages[f"{stage} {encoder.name}"] for stage in ("encode", "search")
            )
        for name, members in fused.items():
            started = time.perf_counter()
            ranked = rrf([rankings[member] for member in members])
            stages[f"fuse {name}"] = time.perf_counter() - started
            rows[f"RRF: {name}"] = sum(rows[member] for member in members) + stages[f"fuse {name}"]
            if reranker is not None:
                started = time.perf_counter()
                rerank(question, ranked, texts, score, args.rerank_depth)
                stages[f"rerank {name}"] = time.perf_counter() - started
                rows[f"RRF: {name}, reranked"] = rows[f"RRF: {name}"] + stages[f"rerank {name}"]
        return stages, rows

    run(questions[0])
    stages: dict[str, list[float]] = {}
    rows: dict[str, list[float]] = {}
    for question in questions:
        for found, times in zip(run(question), (stages, rows), strict=True):
            for name, seconds in found.items():
                times.setdefault(name, []).append(seconds * 1000)

    table(f"row of T4, {len(questions)} questions", rows)
    print()
    table("stage", stages)
    where = device_name(encoders[0][1]) if encoders else device_name(reranker)
    depth = f"; reranking the first {args.rerank_depth}" if reranker is not None else ""
    print(f"\non {where}; {runtime_versions()}; chunks {len(chunks)}{depth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
