#!/usr/bin/env python3
"""Measure BM25 retrieval against the gold set's evidence.

    python3 scripts/eval_retrieval.py [--gold PATH] [--chunks PATH] [options]

Searches the chunks (data/processed/chunks.jsonl, written by daleel chunk)
for every answerable question of the gold set, or of one split of it, and
prints the share of the evidence some chunk holds, recall and completeness at
5 and 10, and MRR@10, overall, by question type and by language. --misses
lists each question whose evidence is not all in the first five, with the
rank of each of its quotes.

The analyzer's steps can be turned off one at a time, and each chunk's heading
indexed with its text, to measure what each contributes. The measures
live in daleel.eval.retrieval and are tested there.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_DRAFT, SPLITS, load_gold
from daleel.eval.retrieval import evaluate, summary
from daleel.retrieve.analyzer import Analyzer
from daleel.retrieve.bm25 import TEXT_ONLY, WITH_HEADING, ChunkIndex

COLUMNS = ("questions", "held", "recall@5", "recall@10", "complete@5", "complete@10", "mrr@10")


def _row(name: str, measures: dict[str, float]) -> str:
    cells = [f"{measures['questions']:>12.0f}"]
    cells += [f"{measures[column]:>12.3f}" for column in COLUMNS[1:]]
    return f"{name:16s}" + "".join(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure BM25 against the gold set.")
    parser.add_argument("--gold", type=Path, default=GOLD_DRAFT)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument(
        "--split", choices=SPLITS, help="only the questions of this split, once there is one"
    )
    parser.add_argument("--no-normalize", action="store_true", help="index the text as extracted")
    parser.add_argument("--no-stopwords", action="store_true", help="keep stopwords")
    parser.add_argument("--no-stem", action="store_true", help="do not stem")
    parser.add_argument("--heading", action="store_true", help="index each chunk's heading too")
    parser.add_argument("--misses", action="store_true", help="list what is not in the top 5")
    parser.add_argument("--json", action="store_true", help="emit the measures as JSON")
    args = parser.parse_args(argv)
    for path in (args.gold, args.chunks):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = [
        question
        for question in load_gold(args.gold)
        if args.split is None or question.get("split") == args.split
    ]
    chunks = read_chunks(args.chunks)
    analyzer = Analyzer(
        normalize=not args.no_normalize, stopwords=not args.no_stopwords, stem=not args.no_stem
    )
    index = ChunkIndex(chunks, analyzer, WITH_HEADING if args.heading else TEXT_ONLY)
    results = evaluate(questions, chunks, index.search)

    groups: dict[str, list] = defaultdict(list)
    for result in results:
        groups[f"type {result.type}"].append(result)
        groups[f"lang {result.lang}"].append(result)
    measures = {"all": summary(results)}
    measures |= {name: summary(found) for name, found in sorted(groups.items())}
    if args.json:
        print(json.dumps(measures, indent=2))
        return 0

    print(f"{len(chunks)} chunks, {len(results)} answerable questions, {analyzer}")
    print(f"{'':16s}" + "".join(f"{column:>12s}" for column in COLUMNS))
    for name, found in measures.items():
        print(_row(name.removeprefix("type ").removeprefix("lang "), found))
    unheld = sum(result.unheld for result in results)
    if unheld:
        print(f"warning: {unheld} quotes are held by no chunk of their page", file=sys.stderr)
    if args.misses:
        print("\nnot all evidence in the first 5:")
        for result in results:
            if not result.complete(5):
                ranks = ", ".join("-" if rank is None else str(rank) for rank in result.ranks)
                print(f"  {result.qid} {result.type:15s} ranks {ranks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
