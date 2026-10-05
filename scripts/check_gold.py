#!/usr/bin/env python3
"""Check a gold question set against its rules.

    python3 scripts/check_gold.py [path]

The path defaults to eval/gold_draft.jsonl. Prints how many questions there
are of each type against the spec, how many a correct answer should answer,
refuse or qualify, how many have been checked by hand, and every problem
found, then exits with 1 if there is any.

This is a validation tool, not part of the package: the checks it prints live
in daleel.eval.gold and are tested there.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from daleel.eval.gold import COMPOSITION, GOLD_DRAFT, behavior, composition, load_gold, problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(args[0]) if args else GOLD_DRAFT
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2

    questions = load_gold(path)
    counts = composition(questions)
    print(f"questions: {len(questions)}")
    for kind, expected in COMPOSITION.items():
        print(f"  {kind:15s} {counts[kind]:3d} (spec {expected})")
    print("what a correct answer does:")
    for name, count in sorted(Counter(behavior(q) for q in questions if _known(q)).items()):
        print(f"  {name:25s} {count:3d}")
    reviewed = sum(question.get("human_reviewed") is True for question in questions)
    print(f"checked by hand against the page: {reviewed} of {len(questions)}")
    found = problems(questions)
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
