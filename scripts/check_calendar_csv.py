#!/usr/bin/env python3
"""Check a hand-transcribed calendar CSV against its rules and the .ics.

    python3 scripts/check_calendar_csv.py <csv> <ics>

Reports the columns, the shape of every date cell, whether each Gregorian date
appears in the .ics, whether each Gregorian range matches one event's start and
end, and whether a row's two calendars agree on how long the event is.

The Hijri column has no external source, so it rests on double annotation plus
that span check. Hijri spans assume 30-day months and allow a day of slack.

This is a validation tool, not part of the package: the checks it prints live
in daleel.eval.calendar_checks and are tested there.
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

from daleel.eval.calendar_checks import ics_spans, read_date_cell, span_disagreement

EXPECTED_COLUMNS = ["title_ar", "title_en", "day_ar", "day_en", "date_gregorian", "date_hijri"]
EXPECTED_ROWS = 14


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    csv_path, ics_path = Path(args[0]), Path(args[1])
    for path in (csv_path, ics_path):
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print("error: the CSV has no rows", file=sys.stderr)
        return 1

    print(f"rows: {len(rows)} (expected {EXPECTED_ROWS})")
    columns = list(rows[0])
    print(f"columns: {'OK' if columns == EXPECTED_COLUMNS else columns}")
    blank = [
        i for i, row in enumerate(rows, start=2) if any(not (v or "").strip() for v in row.values())
    ]
    print(f"rows with an empty cell: {blank or 'none'}")

    days, spans = ics_spans(ics_path.read_bytes())
    malformed, absent, unmatched, disagreeing, checked, skipped = [], [], [], [], 0, 0

    for line_no, row in enumerate(rows, start=2):
        gregorian = read_date_cell(row["date_gregorian"])
        hijri = read_date_cell(row["date_hijri"])
        for column, cell in (("date_gregorian", gregorian), ("date_hijri", hijri)):
            if cell.problem:
                malformed.append((line_no, column, cell.problem))

        problem = span_disagreement(gregorian, hijri)
        if problem:
            disagreeing.append((line_no, row["date_gregorian"], row["date_hijri"], problem))

        if not gregorian.ends:
            skipped += 1
            continue
        dates = [dt.date(*end) for end in gregorian.ends]
        checked += len(dates)
        absent += [
            (line_no, row["date_gregorian"], str(date)) for date in dates if date not in days
        ]
        if gregorian.is_range and frozenset(dates) not in spans:
            unmatched.append(
                (line_no, row["date_gregorian"], "not the start and end of one .ics event")
            )

    print(f"malformed date cells: {malformed or 'none'}")
    print(f"gregorian endpoints checked: {checked} ({skipped} rows skipped as malformed)")
    print(f"dates absent from the .ics: {absent or 'none'}")
    print(f"ranges not matching an .ics event: {unmatched or 'none'}")
    print(f"ranges whose two calendars disagree on length: {disagreeing or 'none'}")
    return 1 if (malformed or absent or unmatched or disagreeing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
