#!/usr/bin/env python3
"""Draw answers to label by hand, for checking the judge (table T9).

    python3 scripts/draw_labels.py --runs RUN [RUN ...] [--count 40] [options]

RUN is a file of answers written by scripts/run_arms.py, such as
data/interim/runs/A-dev.jsonl. Each run gives an equal share of the answers.
Within a run, the questions whose correct answer is something other than a
plain answer (a refusal, a named gap, both sources, the cases told apart) are
drawn first, up to half the run's share, so that every label is likely to
occur, and the rest at random. The draw is seeded, so it can be made again.

Writes to data/interim/labels/:
- sheet.html: the definitions of the labels, then each answer, numbered in a
  shuffled order, with what the judge will read about it. It does not say
  which arm wrote an answer, though one that cites no sources is likely arm
  A's, for the judge as for the person labelling.
- labels.csv: one row per answer, with a label column to fill in: correct,
  partial, incorrect or hallucinated, or their first letters.
- key.jsonl: which run and question each number is. Read it only after
  labelling.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path

from daleel.answer.today import sources_by_id
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.eval.judge import HAND_LABELS, LABELS, POINTS, case

SEED = 1448

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto;
       padding: 0 1rem; line-height: 1.6; color: #1c1c1c; background: #fff; }
section { border-top: 2px solid #999; margin-top: 2.5rem; padding-top: 1rem; }
h3 { font-size: 0.85rem; text-transform: uppercase; color: #555; margin: 1rem 0 0.2rem; }
.text { white-space: pre-wrap; unicode-bidi: plaintext; }
.answer { background: #f4f4f0; padding: 0.6rem 0.8rem; border-radius: 6px; }
dt { font-weight: 700; margin-top: 0.6rem; }
"""


def draw(runs: list[list[dict]], count: int, seed: int = SEED) -> list[dict]:
    """The answers to label, in the order they are shown."""
    rng = random.Random(seed)
    chosen = []
    for place, records in enumerate(runs):
        share = count // len(runs) + (1 if place < count % len(runs) else 0)
        other = [r for r in records if r["behavior"] != "answer"]
        plain = [r for r in records if r["behavior"] == "answer"]
        first = rng.sample(other, min(len(other), share // 2))
        chosen += first + rng.sample(plain, min(len(plain), share - len(first)))
    rng.shuffle(chosen)
    return chosen


def sheet(items: list[dict[str, str]]) -> str:
    """The page a person reads to label the answers."""
    labels = "".join(
        f"<dt>{label}</dt><dd>{html.escape(text)}</dd>" for label, text in LABELS.items()
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Answers to label</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>Answers to label</h1>",
        f"<p>{html.escape(POINTS)}</p><dl>{labels}</dl>",
    ]
    for number, item in enumerate(items, start=1):
        parts.append(f"<section><h2>{number}</h2>")
        for name, text in item.items():
            kind = "text answer" if name == "answer" else "text"
            parts.append(
                f"<h3>{name.replace('_', ' ')}</h3>"
                f"<div class='{kind}' dir='auto'>{html.escape(text)}</div>"
            )
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw answers to label by hand.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--out", type=Path, default=HAND_LABELS)
    args = parser.parse_args(argv)
    for path in (*args.runs, args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = {q["qid"]: q for q in load_gold(args.gold)}
    # The chunks, and the calendar's weeks the flow adds to a question about time.
    chunks = sources_by_id(read_chunks(args.chunks))
    runs = []
    for path in args.runs:
        lines = path.read_text(encoding="utf-8").splitlines()
        runs.append([{**json.loads(line), "run": path.name} for line in lines if line.strip()])
    chosen = draw(runs, args.count, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    items = [case(questions[r["qid"]], r, chunks) for r in chosen]
    (args.out / "sheet.html").write_text(sheet(items), encoding="utf-8")
    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item", "label", "note"])
        writer.writerows([number, "", ""] for number in range(1, len(chosen) + 1))
    (args.out / "key.jsonl").write_text(
        "".join(
            json.dumps({"item": n, "run": r["run"], "arm": r["arm"], "qid": r["qid"]}) + "\n"
            for n, r in enumerate(chosen, start=1)
        ),
        encoding="utf-8",
    )
    arms = {r["arm"]: sum(c["arm"] == r["arm"] for c in chosen) for r in chosen}
    plain = sum(r["behavior"] == "answer" for r in chosen)
    print(
        f"{len(chosen)} answers drawn: " + ", ".join(f"{n} from arm {a}" for a, n in arms.items())
    )
    print(f"{plain} should be answered plainly, {len(chosen) - plain} otherwise")
    print(f"wrote {args.out / 'sheet.html'}, labels.csv to fill in, and key.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
