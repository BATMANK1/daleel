"""Checks for a hand-transcribed calendar CSV.

Ground truth records dates exactly as printed, so no ordering is imposed here:
a printed range may run end first (see rule 6.9 of the annotation
guidelines). What can be checked is the shape of each cell, whether its dates
exist in an independent source, and whether the Gregorian and Hijri ranges
agree on how long the event is. That last check is what catches a cell whose
digits were scrambled while typing mixed-direction text, which no external
source can do for the Hijri column.

Every function here is pure, so the tests need neither a CSV nor a calendar
file.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

# yyyy/mm/dd, optionally -dd or -mm/dd, then the era suffix:
# U+0645 ARABIC LETTER MEEM (Gregorian) or U+0647 ARABIC LETTER HEH (Hijri).
DATE_CELL = re.compile(r"^(\d{4})/(\d{2})/(\d{2})(?:-(?:(\d{2})/)?(\d{2}))?([\u0645\u0647])$")

# Hijri months run 29 or 30 days and this module has no converter, so Hijri
# spans assume 30 and are compared with a tolerance of one day.
HIJRI_MONTH_DAYS = 30
SPAN_TOLERANCE_DAYS = 1

YearMonthDay = tuple[int, int, int]


@dataclass(frozen=True)
class DateCell:
    """A parsed date cell: its one or two ends, in printed order."""

    ends: tuple[YearMonthDay, ...] = ()
    problem: str | None = None

    @property
    def is_range(self) -> bool:
        return len(self.ends) == 2


def read_date_cell(cell: str) -> DateCell:
    """Parse a date cell, or explain why it cannot be read."""
    match = DATE_CELL.fullmatch(cell.strip())
    if not match:
        return DateCell(problem=f"does not match yyyy/mm/dd[-[mm/]dd] plus a suffix: {cell!r}")

    year, month, day, second_month, second_day = (
        int(group) if group else None for group in match.groups()[:5]
    )
    ends: list[YearMonthDay] = [(year, month, day)]
    if second_day:
        ends.append((year, second_month or month, second_day))

    for _, a_month, a_day in ends:
        if not 1 <= a_month <= 12 or not 1 <= a_day <= 31:
            return DateCell(ends=tuple(ends), problem=f"month or day out of range: {cell!r}")
    return DateCell(ends=tuple(ends))


def span_days(cell: DateCell, *, hijri: bool) -> int:
    """How many days a range covers, ignoring which end was printed first.

    Zero for a single date. The Hijri figure is approximate: see
    HIJRI_MONTH_DAYS.
    """
    if not cell.is_range:
        return 0
    if hijri:
        ordinals = [month * HIJRI_MONTH_DAYS + day for _, month, day in cell.ends]
    else:
        ordinals = [dt.date(*end).toordinal() for end in cell.ends]
    return abs(ordinals[1] - ordinals[0])


def span_disagreement(gregorian: DateCell, hijri: DateCell) -> str | None:
    """Report when a row's two calendars disagree on how long the event is."""
    if gregorian.problem or hijri.problem or not (gregorian.is_range and hijri.is_range):
        return None
    in_gregorian = span_days(gregorian, hijri=False)
    in_hijri = span_days(hijri, hijri=True)
    if abs(in_gregorian - in_hijri) <= SPAN_TOLERANCE_DAYS:
        return None
    return f"gregorian spans {in_gregorian} days, hijri about {in_hijri}"


def ics_spans(raw: bytes) -> tuple[set[dt.date], set[frozenset[dt.date]]]:
    """Every event day, and every start-and-end pair, in an iCalendar file.

    Unfolding happens on bytes because a fold can land inside a multi-byte
    character. An all-day DTEND is exclusive, so the real last day is the day
    before the one written.
    """
    text = raw.replace(b"\r\n ", b"").replace(b"\r\n\t", b"").decode("utf-8")
    days: set[dt.date] = set()
    spans: set[frozenset[dt.date]] = set()
    start: dt.date | None = None

    for line in text.split("\r\n"):
        if not line.startswith(("DTSTART", "DTEND")):
            continue
        value = line.split(":", 1)[1]
        date = dt.date(int(value[:4]), int(value[4:6]), int(value[6:8]))
        if line.startswith("DTSTART"):
            start = date
            continue
        if start is None:
            continue
        end = date if "T" in value else date - dt.timedelta(days=1)
        days.update({start, end})
        spans.add(frozenset({start, end}))
        start = None

    return days, spans
