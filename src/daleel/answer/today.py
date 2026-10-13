"""A sense of time: the weeks of each semester, counted from the academic calendar.

The calendar gives the day each semester starts, not week numbers, and no
document knows what day it is. A question about time (which week it is, what
today's date is, how many days are left) is therefore given one more source,
computed here: every week of every semester, counted as the gold set counts
them, from the first day of classes in steps of seven days, holiday weeks
included, each with its Gregorian and Umm al-Qura dates. In the served app
the source opens with today's date in Riyadh, so that an answer about "now"
cites where its date came from.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from itertools import pairwise
from typing import Any

from hijridate import Gregorian

from daleel.corpus import CALENDAR_DOCUMENT
from daleel.normalize.arabic import for_comparison

# Saudi Arabia keeps UTC+3 all year.
RIYADH = timezone(timedelta(hours=3))
WEEKS_ID = "calendar_weeks"
TITLE = "حساب دليل من التقويم الأكاديمي"
# Said to the model with a source that gives today: without it, the model reads the
# date as one more document's claim, and says the documents leave out the current date.
TODAY_NOTE = (
    "Source [{n}] was computed by Daleel from the academic calendar. Its second line "
    "gives today's date, read from the server's clock when the question was asked: it "
    "is the current date. Use it for now, today and this week, and do not say that the "
    "sources leave out the current date."
)
MONTHS = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)
# By date.weekday(): Monday is 0.
DAYS = ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد")

# A semester's first day, as a calendar card gives it: the Gregorian date comes first.
_START = re.compile(r"بداية الفصل[^:]*:[^0-9]*(20\d\d)/(\d{1,2})/(\d{1,2})")
# Questions about time, matched on text normalized for comparison (alef, ya and
# ta marbuta unified, ؟ read as ?).
_ABOUT_TIME = [
    re.compile(pattern)
    for pattern in (
        # Which week: أي أسبوع، باي اسبوع، الأسبوع الكم، هالأسبوع، الأسبوع الحالي.
        r"اي\s*اسبوع(?!ي)",
        r"الاسبوع\s+(ال)?كم",
        r"(هذا|ذا)\s+الاسبوع|هالاسبوع",
        r"الاسبوع\s+(الحالي|الجاي|القادم|الماضي)",
        # How long until: كم باقي على، كم يوم بقي، كم أسبوع على.
        r"كم\s+(يوم|ايام|اسبوع|اسابيع|شهر)\s+(باقي|بقي|متبقي|يبقي|علي|عن|لين|حتي|قبل)",
        r"(باقي|بقي|متبقي)\s+(علي|عن|لين|حتي)",
        # Today's date: تاريخ اليوم، اليوم كم، وش التاريخ.
        r"تاريخ\s+اليوم|اليوم\s+(كم|وش|ايش)|(وش|ايش|كم)\s+(ال)?تاريخ",
    )
]
_ABOUT_TIME_EN = re.compile(
    r"\b(which|what) week\b|\bthis week\b|\bcurrent week\b|\btoday\b|\bdate is it\b"
    r"|\bhow (many|long)\b.*\b(left|until|till|remaining)\b"
)


@dataclass(frozen=True)
class Semester:
    """A semester of the calendar, from its first day up to the next one's."""

    name: str
    start: date
    end: date

    def week(self, day: date) -> int | None:
        """The week of the semester a day falls in, from 1, or None outside it."""
        return (day - self.start).days // 7 + 1 if self.start <= day < self.end else None


def asks_about_time(question: str) -> bool:
    """Whether a question asks which week it is, today's date, or how long until something."""
    text = for_comparison(question)
    return any(pattern.search(text) for pattern in _ABOUT_TIME) or bool(
        _ABOUT_TIME_EN.search(text.lower())
    )


def semesters(chunks: Sequence[Mapping[str, Any]]) -> list[Semester]:
    """The calendar's semesters, each named by its section and ending where the next starts."""
    starts = []
    for chunk in chunks:
        found = _START.search(chunk["text"]) if chunk["doc_id"] == CALENDAR_DOCUMENT else None
        if found:
            year, month, day = (int(part) for part in found.groups())
            starts.append((date(year, month, day), chunk.get("section_heading") or ""))
    starts.sort()
    # The last start (the next year's first semester) only ends the one before it.
    return [Semester(name, start, end) for (start, name), (end, _) in pairwise(starts)]


def spell(day: date) -> str:
    """A day as students write it: the Gregorian date in words and figures, then the
    Umm al-Qura date."""
    hijri = Gregorian(day.year, day.month, day.day).to_hijri()
    return (
        f"{day.day} {MONTHS[day.month - 1]} {day.year}م "
        f"({day:%Y/%m/%d}، {hijri.year}/{hijri.month:02d}/{hijri.day:02d}هـ)"
    )


def time_source(
    chunks: Sequence[Mapping[str, Any]], today: date | None = None
) -> dict[str, Any] | None:
    """The computed source of every semester's weeks, opening with `today` where it is
    given, or None if the chunks hold no calendar."""
    found = semesters(chunks)
    if not found:
        return None
    lines = [
        "أسابيع الفصول الدراسية محسوبة من تاريخ بداية كل فصل في التقويم الأكاديمي: "
        "يبدأ الأسبوع الأول يوم بداية الدراسة، وكل أسبوع سبعة أيام من الأحد إلى السبت، "
        "وتُعد أسابيع الإجازات معها."
    ]
    if today is not None:
        now = f"تاريخ اليوم الحالي، وقت طرح السؤال: {DAYS[today.weekday()]} {spell(today)}."
        current = [(s, s.week(today)) for s in found if s.week(today)]
        lines.append(
            f"{now} الأسبوع الحالي هو الأسبوع {current[0][1]} من {current[0][0].name}."
            if current
            else f"{now} اليوم خارج الفصول الدراسية في التقويم."
        )
    for semester in found:
        weekday = DAYS[semester.start.weekday()]
        lines.append(f"{semester.name}: يبدأ يوم {weekday} {spell(semester.start)}.")
        first, n = semester.start, 1
        while first < semester.end:
            last = first + timedelta(days=6)
            lines.append(f"الأسبوع {n}: من {spell(first)} إلى {spell(last)}")
            first, n = first + timedelta(days=7), n + 1
    calendar = next(chunk for chunk in chunks if chunk["doc_id"] == CALENDAR_DOCUMENT)
    return {
        "chunk_id": WEEKS_ID,
        "doc_id": CALENDAR_DOCUMENT,
        "doc_title": TITLE,
        "scope": calendar.get("scope"),
        "effective": calendar.get("effective"),
        "page": None,
        "section_heading": "أسابيع الفصول الدراسية",
        "clause_no": None,
        "text": "\n".join(lines),
        # Not shown to the model as part of the source; arms.request says what it means.
        **({"today": today.isoformat()} if today is not None else {}),
    }


def sources_by_id(chunks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every source an evaluation run can give the model, by id: the chunks, and the
    calendar's weeks as computed without today."""
    found = {chunk["chunk_id"]: dict(chunk) for chunk in chunks}
    weeks = time_source(chunks)
    if weeks is not None:
        found[WEEKS_ID] = weeks
    return found


def riyadh_today() -> date:
    """Today's date in Riyadh."""
    return datetime.now(UTC).astimezone(RIYADH).date()


def calendar_source(
    chunks: Sequence[Mapping[str, Any]], clock: Callable[[], date] | None = riyadh_today
) -> Callable[[str], dict[str, Any] | None]:
    """For the answering flow: the computed source for a question about time, else None.

    With `clock` None the source leaves out today, so that an evaluation run
    does not depend on the day it is run."""

    def when(question: str) -> dict[str, Any] | None:
        if not asks_about_time(question):
            return None
        return time_source(chunks, clock() if clock is not None else None)

    return when
