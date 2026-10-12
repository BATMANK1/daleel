#!/usr/bin/env python3
"""Tables T5, T7 and T8 from judged runs, with bootstrap intervals.

    python3 scripts/results.py --runs RUN [RUN ...] [--against NAME] [options]

RUN is a file written by scripts/judge_answers.py, such as
data/interim/runs/flow-dev-judged.jsonl. The measures that need no judge
(declining, citations, numbers) are worked out again here from each answer,
so that runs made before a measure changed are counted alike.

T5, per run: accuracy (the share of answers the judge labels correct), with a
95% interval from resampling the questions; correct or partial; numeric
questions answered with the exact number; citation precision against the
gold evidence; answers labelled hallucinated, overall and on questions
outside the corpus; and the difference in accuracy from the run named by
--against (the last run by default), with its interval, the two runs
resampled on the same questions.

T7, per run: the model's seconds at the median and the 95th percentile,
tokens in and out on average, and US dollars per 1,000 questions at paid
rates. Retrieval time is not included; table T4 gives it.

T8, per run: the judge's labels by question type.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from daleel.answer.arms import citations
from daleel.answer.measure import citation_measures, declines, states_number
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import ANSWERABLE, GOLD_V1, load_gold
from daleel.eval.judge import LABELS
from daleel.eval.retrieval import by_page, evidence

DRAWS = 10_000
SEED = 1448


def interval(values: Sequence[float], draws: int = DRAWS, seed: int = SEED) -> tuple[float, float]:
    """The 95% bootstrap interval of a mean, the values resampled with replacement."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def percentile(values: Sequence[float], share: float) -> float:
    """The value below which `share` of the values fall, by the nearest rank."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered)) - 1))]


def measured(records: list[dict], questions: dict, chunks: dict, pages: dict) -> list[dict]:
    """The records with their judge-free measures worked out again."""
    found = []
    for record in records:
        question = questions[record["qid"]]
        sources = [chunks[cid] for cid in record["sources"]]
        cited = citations(record["answer"], len(sources))
        units = evidence(question, pages) if question["answerability_status"] in ANSWERABLE else []
        expected = question.get("answer_numeric")
        found.append(
            {
                **record,
                "declined": declines(record["answer"], cited),
                "numeric_exact": (
                    states_number(record["answer"], expected) if expected is not None else None
                ),
                "precision": (
                    citation_measures(cited, sources, units)["precision"]
                    if units and sources and cited.numbers
                    else None
                ),
            }
        )
    return found


def share(hits: int, total: int) -> str:
    return f"{hits}/{total}" if total else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tables T5, T7 and T8 from judged runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--against", help="the run to compare accuracy with (default: last)")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    args = parser.parse_args(argv)
    for path in (*args.runs, args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = {q["qid"]: q for q in load_gold(args.gold)}
    all_chunks = read_chunks(args.chunks)
    chunks = {chunk["chunk_id"]: chunk for chunk in all_chunks}
    pages = by_page(all_chunks)
    runs: dict[str, list[dict]] = {}
    for path in args.runs:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any("judge" not in record for record in records):
            print(f"error: {path} is not judged; run scripts/judge_answers.py", file=sys.stderr)
            return 2
        runs[path.stem.removesuffix("-judged")] = measured(records, questions, chunks, pages)
    against = args.against or next(reversed(runs))
    if against not in runs:
        print(f"error: no run named {against}; the runs are {', '.join(runs)}", file=sys.stderr)
        return 2
    base = {r["qid"]: r["judge"] == "correct" for r in runs[against]}

    header = ["n", "accuracy", "95% interval", "correct or partial", "numbers exact"]
    header += ["citation precision", "hallucinated", "hallucinated, outside corpus"]
    header += [f"accuracy minus {against}"]
    print("T5 | " + " | ".join(header) + " |")
    print("|---" * (len(header) + 1) + "|")
    for name, records in runs.items():
        right = [float(r["judge"] == "correct") for r in records]
        low, high = interval(right)
        some = sum(r["judge"] in ("correct", "partial") for r in records)
        numeric = [r["numeric_exact"] for r in records if r["numeric_exact"] is not None]
        precision = [r["precision"] for r in records if r["precision"] is not None]
        outside = [r for r in records if r["behavior"] == "refuse"]
        paired = [
            float(r["judge"] == "correct") - base[r["qid"]] for r in records if r["qid"] in base
        ]
        if name == against or len(paired) < len(records):
            difference = "-"
        else:
            d_low, d_high = interval(paired)
            difference = f"{statistics.fmean(paired):+.2f} ({d_low:+.2f} to {d_high:+.2f})"
        cells = [
            str(len(records)),
            f"{statistics.fmean(right):.2f}",
            f"{low:.2f} to {high:.2f}",
            share(some, len(records)),
            share(sum(numeric), len(numeric)),
            f"{statistics.fmean(precision):.2f}" if precision else "-",
            share(sum(r["judge"] == "hallucinated" for r in records), len(records)),
            share(sum(r["judge"] == "hallucinated" for r in outside), len(outside)),
            difference,
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")

    header = ["p50 s", "p95 s", "tokens in", "tokens out", "$/1k"]
    print("\nT7 | " + " | ".join(header) + " |")
    print("|---" * (len(header) + 1) + "|")
    for name, records in runs.items():
        seconds = [r["seconds"] for r in records]
        costs = [r["usd_at_paid_rates"] for r in records if r["usd_at_paid_rates"] is not None]
        cells = [
            f"{statistics.median(seconds):.1f}",
            f"{percentile(seconds, 0.95):.1f}",
            f"{statistics.fmean(r['tokens_in'] for r in records):,.0f}",
            f"{statistics.fmean(r['tokens_out'] for r in records):,.0f}",
            f"{1000 * statistics.fmean(costs):.2f}" if costs else "-",
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")

    for name, records in runs.items():
        print(f"\nT8, {name} | n | " + " | ".join(LABELS) + " | accuracy |")
        print("|---" * (len(LABELS) + 3) + "|")
        for kind in sorted({r["type"] for r in records}):
            mine = [r for r in records if r["type"] == kind]
            found = Counter(r["judge"] for r in mine)
            right = found["correct"] / len(mine)
            labels = " | ".join(str(found[label]) for label in LABELS)
            print(f"| {kind} | {len(mine)} | {labels} | {right:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
