#!/usr/bin/env python3
"""Where to refuse a question without asking the model, set on the dev questions.

    python3 scripts/refusal_threshold.py [options]

For each dev question, takes the first five chunks of the retrieval chosen in
table T4 and the cross-encoder's best score among them, from the vectors and
scores stored in data/interim/, so no GPU is needed. Prints those scores by
the behavior the gold set expects, then what each threshold would refuse,
then the threshold the rule gives: one logit under the lowest score of any
question that should be answered, rounded down to a half.

It reads the dev questions only, so that the final split stays unseen.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from daleel.answer.arms import SOURCES
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, behavior, load_gold
from daleel.retrieve.dense import DENSE
from daleel.retrieve.hybrid import chosen_retrieval
from daleel.retrieve.rerank import REFUSE_BELOW, RERANK, RERANKER_NAME, ScoreStore, best_score

THRESHOLDS = (-5.0, -4.5, -4.0, -3.5, -3.0, -2.5)
MARGIN = 1.0


def rule(lowest: float) -> float:
    """One logit under the lowest score that should be answered, rounded down to a half."""
    return math.floor((lowest - MARGIN) * 2) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set the refusal threshold on dev.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--rerank", type=Path, default=RERANK / f"{RERANKER_NAME}.json")
    args = parser.parse_args(argv)
    for path in (args.chunks, args.gold, args.dense / "bge-m3.npz", args.rerank):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    chunks = read_chunks(args.chunks)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    retrieval = chosen_retrieval(chunks, args.dense, args.rerank)
    best = best_score(ScoreStore.load(args.rerank).score)
    scored = []
    for question in (q for q in load_gold(args.gold) if q.get("split") == "dev"):
        sources = [by_id[cid] for cid in retrieval.search(question["question"], SOURCES)]
        scored.append((best(question["question"], sources), behavior(question), question["qid"]))

    print(
        f"Best cross-encoder score among the first {SOURCES} sources, {len(scored)} dev questions:"
    )
    print("\n| expected behavior | questions | lowest | median | highest |\n|---|---|---|---|---|")
    for expected in sorted({b for _, b, _ in scored}):
        values = [score for score, b, _ in scored if b == expected]
        print(
            f"| {expected} | {len(values)} | {min(values):.2f} | "
            f"{statistics.median(values):.2f} | {max(values):.2f} |"
        )

    answer = [score for score, b, _ in scored if b != "refuse"]
    outside = [score for score, b, _ in scored if b == "refuse"]
    print("\n| refuse below | should answer, refused | outside corpus, refused |\n|---|---|---|")
    for threshold in THRESHOLDS:
        wrong = sum(score < threshold for score in answer)
        right = sum(score < threshold for score in outside)
        print(f"| {threshold:.1f} | {wrong}/{len(answer)} | {right}/{len(outside)} |")

    print("\nLowest five:")
    for score, expected, qid in sorted(scored)[:5]:
        print(f"  {qid} {expected} {score:.2f}")
    found = rule(min(answer))
    print(
        f"\nThe rule gives {found:.1f}: the lowest score that should be answered, "
        f"{min(answer):.2f}, less {MARGIN:.0f}, rounded down to a half."
    )
    if found != REFUSE_BELOW:
        print(f"REFUSE_BELOW is {REFUSE_BELOW}, not {found}: update it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
