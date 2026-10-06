#!/usr/bin/env python3
"""Check a gold question set against its rules, and its quotes against the pages.

    python3 scripts/check_gold.py [path] [--records DIR]

The path defaults to eval/gold_v1.jsonl. Prints how many questions there
are of each type against the set's composition, how many a correct answer
should answer, refuse or qualify, how many are answered and how many checked
by hand, and every problem found, then exits with 1 if there is any.

With --records, the extracted records (data/interim/extracted/ by default
when the option is given without a folder) are read too, and every quote must
be found in the text of the page it cites.

This is a validation tool, not part of the package: the checks it prints live
in daleel.eval.gold and are tested there.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from daleel.eval.gold import (
    COMPOSITION,
    GOLD_V1,
    behavior,
    composition,
    evidence_problems,
    load_gold,
    problems,
)
from daleel.ingest.records import RECORDS, read_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a gold question set.")
    parser.add_argument("path", nargs="?", type=Path, default=GOLD_V1)
    parser.add_argument("--records", nargs="?", type=Path, const=RECORDS, default=None)
    args = parser.parse_args(argv)
    path, records = args.path, args.records
    for needed in [path] + ([records] if records else []):
        if not needed.exists():
            print(f"error: {needed} does not exist", file=sys.stderr)
            return 2

    questions = load_gold(path)
    counts = composition(questions)
    print(f"questions: {len(questions)}")
    for kind, expected in COMPOSITION.items():
        print(f"  {kind:15s} {counts[kind]:3d} (of {expected})")
    print("what a correct answer does:")
    for name, count in sorted(Counter(behavior(q) for q in questions if _known(q)).items()):
        print(f"  {name:25s} {count:3d}")
    answered = sum(bool(question.get("answer")) for question in questions)
    reviewed = sum(question.get("human_reviewed") is True for question in questions)
    print(f"answered: {answered} of {len(questions)}")
    print(f"checked by hand against the page: {reviewed} of {len(questions)}")
    splits = Counter(question.get("split") for question in questions)
    if any(splits):
        print(f"split: {splits['dev']} for development, {splits['final']} held out")
    found = problems(questions)
    if records:
        pages = {
            (record["doc_id"], record["page"]): record.get("text") or ""
            for file in sorted(records.glob("*.jsonl"))
            for record in read_records(file)
        }
        found += evidence_problems(questions, pages)
        print(f"quotes checked against {len(pages)} pages in {records}")
    print(f"problems: {len(found) or 'none'}")
    for problem in found:
        print(f"  {problem}")
    return 1 if found else 0


def _known(question: dict) -> bool:
    try:
        behavior(question)
    except KeyError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
