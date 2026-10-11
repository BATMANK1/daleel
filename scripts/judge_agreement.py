#!/usr/bin/env python3
"""Table T9: the judge's labels against labels made by hand.

    python3 scripts/judge_agreement.py [--labels data/interim/labels] [options]

Reads the hand labels in labels.csv, which scripts/draw_labels.py wrote with
the sheet, and key.jsonl, which says which answer each item is. Asks the
judge (daleel.eval.judge) for its label on each answer, through
daleel.answer.llm, which stores the replies. Prints the confusion table,
hand labels by row and the judge's by column, Cohen's kappa, agreement by
arm, and every item where the two differ with the judge's reasoning, for
eval/judge_agreement.md. The judged items go to judged.jsonl beside the
labels.

Refuses a labels file with an item unlabelled or labelled with anything but
correct, partial, incorrect or hallucinated (or their first letters), so that
the judge is only asked once the hand labels are final.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from daleel.answer.llm import JUDGE, Client, QuotaExhaustedError
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.agreement import cohen_kappa, confusion
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.eval.judge import HAND_LABELS, LABELS, RUBRIC_VERSION, hand_label, judge

RUNS = Path("data/interim/runs")


def read_labels(path: Path) -> tuple[dict[int, str], dict[int, str], list[str]]:
    """The hand labels and notes by item, and the items that cannot be read."""
    labels, notes, problems = {}, {}, []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            item = int(row["item"])
            label = hand_label(row.get("label") or "")
            if label is None:
                problems.append(f"item {item}: {row.get('label')!r}")
            else:
                labels[item] = label
            notes[item] = (row.get("note") or "").strip()
    return labels, notes, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Table T9: the judge against hand labels.")
    parser.add_argument("--labels", type=Path, default=HAND_LABELS)
    parser.add_argument("--runs", type=Path, default=RUNS, help="the folder of the answer files")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--reasoning", type=int, default=600, help="characters of it to print")
    args = parser.parse_args(argv)
    sheet, key_file = args.labels / "labels.csv", args.labels / "key.jsonl"
    for path in (sheet, key_file, args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    human, notes, problems = read_labels(sheet)
    key = [json.loads(line) for line in key_file.read_text(encoding="utf-8").splitlines()]
    problems += [f"item {k['item']}: no row" for k in key if k["item"] not in notes]
    if problems:
        print("error: label every item first:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 4

    questions = {q["qid"]: q for q in load_gold(args.gold)}
    chunks = {chunk["chunk_id"]: chunk for chunk in read_chunks(args.chunks)}
    runs: dict[str, dict[str, dict]] = {}
    for name in {k["run"] for k in key}:
        lines = (args.runs / name).read_text(encoding="utf-8").splitlines()
        runs[name] = {r["qid"]: r for r in map(json.loads, lines)}
    client = Client()
    judged = []
    try:
        for k in key:
            record = runs[k["run"]][k["qid"]]
            found = judge(questions[k["qid"]], record, chunks, client.ask)
            judged.append({**k, "human": human[k["item"]], "note": notes[k["item"]], **found})
            print(f"judged {len(judged)} of {len(key)}", end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    except QuotaExhaustedError as error:
        print(f"stopped: {error}. Verdicts so far are stored; run again later.", file=sys.stderr)
        return 3
    (args.labels / "judged.jsonl").write_text(
        "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judged), encoding="utf-8"
    )

    pairs = [j for j in judged if j["judge"] is not None]
    order = list(LABELS)
    table = confusion([j["human"] for j in pairs], [j["judge"] for j in pairs], order)
    print(f"| {len(pairs)} answers | " + " | ".join(f"judge: {o}" for o in order) + " |")
    print("|---" * (len(order) + 1) + "|")
    for label, row in zip(order, table, strict=True):
        print(f"| human: {label} | " + " | ".join(map(str, row)) + " |")
    agree = sum(j["human"] == j["judge"] for j in pairs)
    kappa = cohen_kappa([j["human"] for j in pairs], [j["judge"] for j in pairs], order)
    print(f"\nCohen's kappa {kappa:.2f}; the labels agree on {agree} of {len(pairs)}.")
    print(f"Judge {JUDGE}, rubric {RUBRIC_VERSION}.")
    if len(pairs) < len(judged):
        print(f"{len(judged) - len(pairs)} answers had no verdict from the judge.")
    for arm in sorted({j["arm"] for j in pairs}):
        mine = [j for j in pairs if j["arm"] == arm]
        print(f"arm {arm}: agree on {sum(j['human'] == j['judge'] for j in mine)} of {len(mine)}")

    print("\nDisagreements:")
    for j in judged:
        if j["judge"] != j["human"]:
            print(
                f"\n- item {j['item']} ({j['qid']}, arm {j['arm']}): human {j['human']}, "
                f"judge {j['judge']}" + (f". Note: {j['note']}" if j["note"] else "")
            )
            print("  " + j["judge_reasoning"][-args.reasoning :].replace("\n", "\n  "))
    print(f"\n{client.asked} requests sent now; the rest came from the store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
