#!/usr/bin/env python3
"""Judge every answer of a run, for accuracy (tables T5 and T8).

    python3 scripts/judge_answers.py --runs RUN [RUN ...] [options]

RUN is a file of answers written by scripts/run_arms.py. Each answer is
labelled by the judge (daleel.eval.judge) through daleel.answer.llm, which
stores the replies, and the labelled answers go to <run>-judged.jsonl beside
the run. Trust the labels only once table T9 has validated the judge
(scripts/judge_agreement.py).

Prints one row per run: how many answers got each label, how many got no
verdict, accuracy (the share labelled correct), and how many questions
outside the corpus were answered with invented details. Then, for each run,
the labels by question type.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from daleel.answer.llm import Client, QuotaExhaustedError
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.eval.judge import LABELS, judge


def judged_name(run: Path) -> Path:
    return run.with_name(f"{run.stem}-judged.jsonl")


def counts(records: list[dict]) -> list[str]:
    """How many answers got each label, then none."""
    found = Counter(r["judge"] for r in records)
    return [str(found[label]) for label in LABELS] + [str(found[None])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge every answer of a run.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    args = parser.parse_args(argv)
    for path in (*args.runs, args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = {q["qid"]: q for q in load_gold(args.gold)}
    chunks = {chunk["chunk_id"]: chunk for chunk in read_chunks(args.chunks)}
    client = Client()
    judged: dict[str, list[dict]] = {}
    for run in args.runs:
        records = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
        try:
            found = []
            for r in records:
                found.append({**r, **judge(questions[r["qid"]], r, chunks, client.ask)})
                print(
                    f"{run.stem}: judged {len(found)} of {len(records)}",
                    end="\r",
                    file=sys.stderr,
                    flush=True,
                )
            print(file=sys.stderr)
        except QuotaExhaustedError as error:
            print(
                f"stopped: {error}. Verdicts so far are stored; run again later.", file=sys.stderr
            )
            return 3
        judged_name(run).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in found), encoding="utf-8"
        )
        judged[run.stem] = found

    header = ["answers", *LABELS, "no verdict", "accuracy", "hallucinated, outside corpus"]
    print("| run | " + " | ".join(header) + " |")
    print("|---" * (len(header) + 1) + "|")
    for name, records in judged.items():
        outside = [r for r in records if r["behavior"] == "refuse"]
        invented = sum(r["judge"] == "hallucinated" for r in outside)
        accuracy = sum(r["judge"] == "correct" for r in records) / len(records)
        cells = [
            str(len(records)),
            *counts(records),
            f"{accuracy:.2f}",
            f"{invented}/{len(outside)}",
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")

    for name, records in judged.items():
        print(f"\n| {name}, by type | n | " + " | ".join([*LABELS, "no verdict"]) + " |")
        print("|---" * (len(LABELS) + 3) + "|")
        for kind in sorted({r["type"] for r in records}):
            mine = [r for r in records if r["type"] == kind]
            print(f"| {kind} | {len(mine)} | " + " | ".join(counts(mine)) + " |")
    print(f"\n{client.asked} requests sent now; the rest came from the store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
