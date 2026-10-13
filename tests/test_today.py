"""Tests for the calendar's weeks and today's date, given to questions about time."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from daleel.answer.today import (
    TITLE,
    WEEKS_ID,
    Semester,
    asks_about_time,
    calendar_source,
    semesters,
    sources_by_id,
    spell,
    time_source,
)
from daleel.corpus import CALENDAR_DOCUMENT

# The letter heh that marks a Hijri date, alone as the calendar prints it, and
# with its stroke as a source spells it.
H = chr(0x0647)
HIJRI = H + chr(0x0640)


def card(n: int, heading: str, text: str) -> dict:
    """A calendar card as the chunker writes it: its section, then the entry."""
    return {
        "chunk_id": f"{CALENDAR_DOCUMENT}_p1_c{n}",
        "doc_id": CALENDAR_DOCUMENT,
        "doc_title": "التقويم الأكاديمي للعام الدراسي 1448 - 1449 هـ",
        "scope": "commission",
        "effective": "1448-1449 AH / 2026-2027",
        "page": 1,
        "section_heading": heading,
        "clause_no": None,
        "text": f"{heading}، {text}",
    }


FIRST, SECOND, SUMMER = (
    "الفصل الدراسي الأول (481)",
    "الفصل الدراسي الثاني (482)",
    "الفصل الدراسي الصيفي (483)",
)
CALENDAR = [
    card(
        1,
        FIRST,
        "بداية الفصل الدراسي الأول (Start of First Semester): الأحد (Sun) "
        f"2026/08/23م، 1448/03/10{H}",
    ),
    card(
        6,
        FIRST,
        "بداية الاختبارات النصفية (Start of Midterm Exams): الأحد (Sun) "
        f"2026/10/11م، 1448/04/30{H}",
    ),
    card(
        14,
        SECOND,
        "بداية الفصل الدراسي الثاني (Start of Second Semestet): الأحد (Sun) 2027/01/17 م، "
        f"1448/08/09{H}",
    ),
    card(
        20,
        SUMMER,
        f"بداية الفصل الصيفي (Start of Summer Semester): الأحد (Sun) 2027/06/20 م، 1449/01/15 {H}",
    ),
    card(
        28,
        SUMMER,
        "بداية الفصل الدراسي الأول (491) (Start of First Semester (491)): الأحد (Sun) "
        f"2027/08/22م، 1449/03/20{H}",
    ),
]
GUIDE = {
    "chunk_id": "guide_p30_c1",
    "doc_id": "student_guide_2025",
    "doc_title": "دليل الطالب",
    "page": 30,
    "section_heading": "",
    "clause_no": None,
    "text": "يحرم الطالب إذا تجاوز غيابه 20% وبداية الفصل الدراسي تحددها الكلية: 2026/01/01",
}


@pytest.mark.parametrize(
    "question",
    [
        "احنا الاسبوع الكم؟",
        "بتاريخ 13 اكتوبر احنا باي اسبوع بالترم الاول",
        "أي أسبوع نحن فيه؟",
        "وش الأسبوع الحالي من الفصل؟",
        "كم باقي على الاختبارات النهائية؟",
        "كم يوم بقي على إجازة الخريف",
        "وش تاريخ اليوم؟",
        "Which week of the semester is it?",
        "How many days are left until the final exams?",
    ],
)
def test_a_question_about_time_is_recognized(question: str) -> None:
    assert asks_about_time(question)


@pytest.mark.parametrize(
    "question",
    [
        "لمن أرسل التقارير الأسبوعية والتقرير النهائي للتدريب التعاوني؟",
        "لو المقرر 3 ساعات أسبوعياً وعدد أسابيع تدريسه 15، كم ساعة غياب تساوي 20%؟",
        "إذا انسحبت في الأسبوع الثاني أو الرابع، كم أسترد من الرسوم؟",
        "إجازة اليوم الوطني في الفصل الأول كم يوم؟",
        "أي الحدين ينطبق الآن على طلاب ينبع؟",
        "متى تبدأ الاختبارات النهائية في الفصل الأول؟",
        "What is the attendance rule for exams?",
    ],
)
def test_a_question_about_the_rules_is_not_taken_for_one_about_time(question: str) -> None:
    assert not asks_about_time(question)


def test_the_gold_set_s_questions_about_time_are_its_three_about_weeks() -> None:
    gold = Path("eval/gold_v1.jsonl")
    questions = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    found = {q["qid"] for q in questions if asks_about_time(q["question"])}
    assert found == {"g028", "g029", "g033"}


def test_each_semester_runs_from_its_first_day_to_the_next_one_s() -> None:
    assert semesters([GUIDE, *CALENDAR]) == [
        Semester(FIRST, date(2026, 8, 23), date(2027, 1, 17)),
        Semester(SECOND, date(2027, 1, 17), date(2027, 6, 20)),
        Semester(SUMMER, date(2027, 6, 20), date(2027, 8, 22)),
    ]
    assert semesters([GUIDE]) == []


@pytest.mark.parametrize(
    ("day", "week"),
    [
        (date(2026, 8, 23), 1),
        (date(2026, 8, 29), 1),
        (date(2026, 8, 30), 2),
        # The gold set's g028: 4 October 2026 begins the seventh week.
        (date(2026, 10, 4), 7),
        (date(2026, 10, 13), 8),
        (date(2027, 1, 16), 21),
        (date(2027, 1, 17), None),
        (date(2026, 8, 22), None),
    ],
)
def test_weeks_are_counted_in_sevens_from_the_first_day(day: date, week: int | None) -> None:
    assert Semester(FIRST, date(2026, 8, 23), date(2027, 1, 17)).week(day) == week


def test_a_day_is_spelled_with_its_gregorian_and_umm_al_qura_dates() -> None:
    # The calendar's own Hijri dates for these days.
    assert spell(date(2026, 8, 23)) == f"23 أغسطس 2026م (2026/08/23، 1448/03/10{HIJRI})"
    assert spell(date(2026, 10, 11)) == f"11 أكتوبر 2026م (2026/10/11، 1448/04/30{HIJRI})"


def test_the_source_lists_every_week_and_opens_with_today() -> None:
    source = time_source([GUIDE, *CALENDAR], today=date(2026, 10, 13))
    assert source is not None
    assert (source["chunk_id"], source["doc_id"], source["doc_title"]) == (
        WEEKS_ID,
        CALENDAR_DOCUMENT,
        TITLE,
    )
    assert source["page"] is None
    lines = source["text"].splitlines()
    assert lines[1] == (
        f"اليوم الثلاثاء 13 أكتوبر 2026م (2026/10/13، 1448/05/02{HIJRI})، "
        "في الأسبوع 8 من الفصل الدراسي الأول (481)."
    )
    assert (
        f"الأسبوع 8: من 11 أكتوبر 2026م (2026/10/11، 1448/04/30{HIJRI}) "
        f"إلى 17 أكتوبر 2026م (2026/10/17، 1448/05/06{HIJRI})"
    ) in lines
    # 21 weeks of the first semester, 22 of the second, 9 of the summer.
    assert sum(line.startswith("الأسبوع ") for line in lines) == 21 + 22 + 9


def test_without_today_the_source_is_the_same_every_day() -> None:
    source = time_source(CALENDAR)
    assert source is not None and "اليوم" not in source["text"].splitlines()[1]
    assert time_source([GUIDE]) is None


def test_a_day_between_semesters_is_said_to_be_outside_them() -> None:
    source = time_source(CALENDAR, today=date(2027, 9, 1))
    assert source is not None
    assert source["text"].splitlines()[1].endswith("خارج الفصول الدراسية في التقويم.")


def test_the_flow_s_hook_adds_the_source_to_questions_about_time_alone() -> None:
    when = calendar_source(CALENDAR, clock=lambda: date(2026, 10, 13))
    added = when("احنا الاسبوع الكم؟")
    assert added is not None and "الأسبوع 8 من" in added["text"]
    assert when("كم رسوم الوحدة الدراسية؟") is None
    unchanging = calendar_source(CALENDAR, clock=None)("احنا الاسبوع الكم؟")
    assert unchanging == time_source(CALENDAR)


def test_an_evaluation_finds_the_computed_source_by_its_id() -> None:
    found = sources_by_id([GUIDE, *CALENDAR])
    assert found[WEEKS_ID] == time_source(CALENDAR) and found["guide_p30_c1"] == GUIDE
    assert WEEKS_ID not in sources_by_id([GUIDE])
