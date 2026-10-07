#!/usr/bin/env python3
"""Table T4: BM25, dense encoders, their fusion and reranking, against the gold set.

    python3 scripts/eval_hybrid.py [--split dev] [--rerank-depth 20] [--json]

Reads the chunks, the dense vectors in data/interim/dense/
(scripts/encode_dense.py) and the cross-encoder's scores in
data/interim/rerank/ (scripts/score_rerank_pool.py), so it runs without the
models or a GPU. Its rows:

- BM25 with the full analyzer, as in table T3;
- each dense encoder alone;
- BM25 fused with each encoder, and with both, by reciprocal rank fusion;
- each fusion with its first chunks reranked by the cross-encoder.

Rows whose vectors or scores are missing are left out, and the output says
which. Prints a Markdown table of the measures of daleel.eval.retrieval,
with recall@5 over the English questions alone in the last column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, SPLITS, load_gold
from daleel.eval.retrieval import evaluate, summary
from daleel.retrieve.bm25 import ChunkIndex
from daleel.retrieve.dense import DENSE, ENCODERS, DenseIndex, VectorStore
from daleel.retrieve.hybrid import Hybrid, fusions
from daleel.retrieve.rerank import RERANK, RERANK_DEPTH, RERANKER_NAME, ScoreStore

COLUMNS = ("held", "recall@5", "recall@10", "complete@5", "mrr@10")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Table T4: hybrid retrieval.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--rerank", type=Path, default=RERANK / f"{RERANKER_NAME}.json")
    parser.add_argument("--rerank-depth", type=int, default=RERANK_DEPTH)
    parser.add_argument("--json", action="store_true", help="emit the measures as JSON")
    args = parser.parse_args(argv)
    for path in (args.gold, args.chunks):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = [q for q in load_gold(args.gold) if q.get("split") == args.split]
    chunks = read_chunks(args.chunks)
    texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    bm25 = ChunkIndex(chunks)
    dense = {}
    for name in sorted(ENCODERS):
        path = args.dense / f"{name}.npz"
        if path.exists():
            dense[name] = DenseIndex(chunks, VectorStore.load(path))
        else:
            print(f"note: no {path}, so no rows for {name}", file=sys.stderr)
    scores = ScoreStore.load(args.rerank) if args.rerank.exists() else None
    if scores is None:
        print(f"note: no {args.rerank}, so no reranked rows", file=sys.stderr)

    rows: dict[str, Hybrid] = {"BM25": Hybrid([bm25.search])}
    for name, index in dense.items():
        rows[name] = Hybrid([index.search])
    fused = fusions(bm25.search, {name: index.search for name, index in dense.items()})
    for name, searches in fused.items():
        rows[f"RRF: {name}"] = Hybrid(searches)
    if scores is not None:
        for name, searches in fused.items():
            rows[f"RRF: {name}, reranked"] = Hybrid(
                searches, scores.score, texts, rerank_depth=args.rerank_depth
            )

    measures = {}
    for name, hybrid in rows.items():
        try:
            results = evaluate(questions, chunks, hybrid.search)
        except KeyError as error:
            print(f"error: {name}: {error.args[0]}", file=sys.stderr)
            return 1
        found = summary(results)
        english = [result for result in results if result.lang == "en"]
        found["english recall@5"] = summary(english)["recall@5"] if english else float("nan")
        measures[name] = found
    if args.json:
        print(json.dumps(measures, indent=2))
        return 0

    asked = next(iter(measures.values()))["questions"]
    header = [
        f"{asked:.0f} answerable questions of the {args.split} split",
        *COLUMNS,
        "english R@5",
    ]
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for name, found in measures.items():
        cells = [f"{found[column]:.3f}" for column in COLUMNS]
        print(f"| {name} | " + " | ".join(cells) + f" | {found['english recall@5']:.3f} |")
    if scores is not None:
        print(f"\nreranked: the first {args.rerank_depth} by {scores.meta.get('model', '?')}")
    for name in dense:
        meta = VectorStore.load(args.dense / f"{name}.npz").meta
        print(f"{name}: {meta.get('model')} at {meta.get('revision')}, on {meta.get('device')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
