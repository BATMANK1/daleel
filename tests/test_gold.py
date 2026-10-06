"""Tests for the gold question set and its checks.

The working set is in the repository, so the first tests check it as it
stands: CI fails if an edit breaks it. The rest build one valid question and
break it a field at a time. Whether each quote is on its page needs the
extracted records, which CI does not have, so that check is tested on page
texts made up for it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from daleel.eval.gold import (
    COMPOSITION,
    CORPUS,
    FINAL_SIZE,
    GOLD_DRAFT,
    behavior,
    composition,
    evidence_problems,
    fragments,
    load_gold,
    problems,
    question_problems,
    split_problems,
)

ROOT = Path(__file__).parent.parent
DRAFT = load_gold(ROOT / GOLD_DRAFT)


def test_the_working_set_passes_every_check() -> None:
    assert problems(DRAFT) == []


def test_the_working_set_has_the_spec_s_composition() -> None:
    assert composition(DRAFT) == COMPOSITION
    assert len(DRAFT) == sum(COMPOSITION.values()) == 80


def test_every_question_in_the_working_set_has_an_answer() -> None:
    assert all(question.get("answer") for question in DRAFT)


def test_three_questions_are_refused_and_five_answered_in_part() -> None:
    by_qid = {question["qid"]: behavior(question) for question in DRAFT}
    assert [qid for qid, b in sorted(by_qid.items()) if b == "refuse"] == ["g074", "g075", "g076"]
    assert [qid for qid, b in sorted(by_qid.items()) if b == "answer_and_name_the_gap"] == [
        "g069",
        "g070",
        "g071",
        "g072",
        "g073",
    ]


def test_the_corpus_is_the_one_data_readme_lists() -> None:
    table = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    listed = {
        name: int(pages) for name, pages in re.findall(r"\| `(\w+)\.pdf` \|[^|]*\| (\d+) \|", table)
    }
    assert listed == CORPUS


def question(**changes: object) -> dict:
    """A valid answerable question, changed as given."""
    valid = {
        "qid": "g001",
        "lang": "ar",
        "type": "single_clause",
        "question": "متى آخر يوم للحذف والإضافة؟",
        "topic_group": "calendar_481_add_drop",
        "answerable": True,
        "answerability_status": "supported",
        "source_refs": [
            {"doc_id": "academic_weeks_1448", "pdf_page": 1, "role": "answer_evidence"},
        ],
        "human_reviewed": False,
        "split": None,
    }
    return valid | changes


def test_a_valid_question_has_no_problems() -> None:
    assert question_problems(question()) == []


ELSEWHERE = {"doc_id": "student_charter", "pdf_page": 4, "role": "related_only"}


@pytest.mark.parametrize(
    ("changes", "complaint"),
    [
        ({"qid": "q1"}, "an id is g and three digits"),
        ({"type": "trivia"}, "unknown type"),
        ({"lang": "fr"}, "language"),
        ({"question": " "}, "no question"),
        ({"answerability_status": "maybe"}, "unknown answerability"),
        ({"answerable": False}, "answerable is False"),
        ({"split": "test"}, "split"),
        ({"type": "english_query"}, "English query in 'ar'"),
        (
            {
                "source_refs": [
                    {"doc_id": "student_handbook", "pdf_page": 1, "role": "answer_evidence"}
                ]
            },
            "not in the corpus",
        ),
        (
            {
                "source_refs": [
                    {"doc_id": "student_charter", "pdf_page": 9, "role": "answer_evidence"}
                ]
            },
            "which has 8 pages",
        ),
        (
            {"source_refs": [{"doc_id": "student_charter", "pdf_page": 4, "role": "evidence"}]},
            "role",
        ),
        ({"source_refs": [ELSEWHERE]}, "no reference holds the answer"),
        ({"source_refs": []}, "no reference holds the answer"),
        ({"type": "cross_document"}, "answered from one document"),
        ({"type": "out_of_scope"}, "type out_of_scope does not fit status supported"),
        ({"type": "precedence"}, "type precedence does not fit status supported"),
        ({"answers": "2026/08/27"}, "unknown field answers"),
    ],
    ids=[
        "bad_id",
        "unknown_type",
        "unknown_language",
        "blank_question",
        "unknown_status",
        "answerable_contradicts_status",
        "unknown_split",
        "english_query_in_arabic",
        "unknown_document",
        "page_past_the_end",
        "unknown_role",
        "only_related_references",
        "no_references",
        "cross_document_from_one",
        "out_of_scope_but_answerable",
        "precedence_without_a_conflict",
        "unknown_field",
    ],
)
def test_each_broken_field_is_reported(changes: dict, complaint: str) -> None:
    found = question_problems(question(**changes))
    assert any(complaint in problem for problem in found), found


def test_a_missing_field_is_reported() -> None:
    broken = question()
    del broken["topic_group"]
    assert question_problems(broken) == ["g001: missing topic_group"]


def test_a_question_outside_the_corpus_needs_no_reference() -> None:
    refused = question(
        type="out_of_scope",
        answerable=False,
        answerability_status="outside_corpus",
        source_refs=[],
    )
    assert question_problems(refused) == []
    assert behavior(refused) == "refuse"


def test_a_partial_answer_cites_only_related_pages() -> None:
    partial = question(
        type="out_of_scope",
        answerable=False,
        answerability_status="unsupported_detail",
        source_refs=[ELSEWHERE | {"role": "answer_evidence"}],
    )
    assert any("marked as holding the answer" in p for p in question_problems(partial))
    assert question_problems(partial | {"source_refs": [ELSEWHERE]}) == []


QUOTED = {
    "doc_id": "academic_weeks_1448",
    "pdf_page": 1,
    "role": "answer_evidence",
    "quotes": [["نهاية فترة حذف وإضافة المقررات", "2026/08/27"]],
}


def answered(**changes: object) -> dict:
    """A valid question with its answer and quotes, changed as given."""
    return question(answer="الخميس 2026/08/27م", source_refs=[QUOTED]) | changes


def test_an_answered_question_with_quotes_has_no_problems() -> None:
    assert question_problems(answered()) == []
    assert question_problems(answered(answer_numeric=27, notes="from the calendar")) == []


@pytest.mark.parametrize(
    ("changes", "complaint"),
    [
        ({"answer": " "}, "an empty answer"),
        ({"answer_numeric": "27"}, "answer_numeric is not a number"),
        ({"answer_numeric": True}, "answer_numeric is not a number"),
        ({"source_refs": [QUOTED | {"quotes": []}]}, "quotes is not a list of quotes"),
        ({"source_refs": [QUOTED | {"quotes": [["", "27"]]}]}, "a quote is empty"),
        ({"source_refs": [QUOTED | {"quotes": [7]}]}, "neither a string nor a list"),
        (
            {"source_refs": [QUOTED, ELSEWHERE | {"quotes": ["حقوق الطالب"]}]},
            "does not hold the answer",
        ),
        ({"source_refs": [{k: v for k, v in QUOTED.items() if k != "quotes"}]}, "no quote from"),
        ({"source_refs": [QUOTED | {"page": 1}]}, "unknown field page"),
    ],
    ids=[
        "empty_answer",
        "number_as_text",
        "true_is_not_a_number",
        "no_quotes",
        "empty_quote",
        "quote_of_the_wrong_kind",
        "quote_on_a_related_page",
        "evidence_without_quotes",
        "unknown_reference_field",
    ],
)
def test_each_broken_answer_is_reported(changes: dict, complaint: str) -> None:
    found = question_problems(answered(**changes))
    assert any(complaint in problem for problem in found), found


def test_a_question_checked_by_hand_has_an_answer() -> None:
    assert question_problems(question(human_reviewed=True)) == [
        "g001: checked by hand, but there is no answer"
    ]


def test_quote_groups_open_up() -> None:
    assert fragments(["a", ["b", "c"]]) == ["a", "b", "c"]


# A calendar card as the text layer gives it: the suffix with its tatweel,
# and alef with hamza where the quote writes a bare one. The suffixes are built
# from their code points, since a lone heh beside digits reads as a Latin o.
MEEM, HEH, TATWEEL = chr(0x0645), chr(0x0647), chr(0x0640)
CARD = "\n".join(
    [
        "نهاية فترة حذف وإضافة المقررات",
        "Last day for add / drop courses",
        "2026/08/27" + MEEM,
        "1448/03/14" + HEH + TATWEEL,
    ]
)


def test_quotes_are_found_on_their_page_once_both_are_normalized() -> None:
    quoted = answered(
        source_refs=[QUOTED | {"quotes": ["نهاية فترة حذف واضافة", "1448/03/14" + HEH]}]
    )
    assert evidence_problems([quoted], {("academic_weeks_1448", 1): CARD}) == []


def test_a_quote_the_page_does_not_hold_is_reported() -> None:
    quoted = answered(source_refs=[QUOTED | {"quotes": ["بداية الفصل الدراسي الأول"]}])
    assert evidence_problems([quoted], {("academic_weeks_1448", 1): CARD}) == [
        "g001: academic_weeks_1448 page 1 does not hold 'بداية الفصل الدراسي الأول'"
    ]


def test_a_page_without_text_is_reported() -> None:
    assert evidence_problems([answered()], {}) == ["g001: no text for academic_weeks_1448 page 1"]


def test_repeated_ids_and_a_wrong_composition_are_reported() -> None:
    found = problems([question(), question()])
    assert "repeated ids: g001" in found
    assert "2 single_clause questions, not 25" in found


def _split(qids_final: set[int], groups: dict[int, str] | None = None) -> list[dict]:
    groups = groups or {}
    return [
        question(
            qid=f"g{n:03d}",
            topic_group=groups.get(n, f"group_{n}"),
            split="final" if n in qids_final else "dev",
        )
        for n in range(1, 81)
    ]


def test_a_split_keeps_each_topic_group_whole() -> None:
    assert split_problems(_split(set(range(1, FINAL_SIZE + 1)))) == []
    divided = _split(set(range(1, FINAL_SIZE + 1)), groups={20: "add_drop", 21: "add_drop"})
    assert split_problems(divided) == ["topic group add_drop is divided between dev, final"]


def test_a_split_holds_out_exactly_twenty() -> None:
    assert split_problems(_split(set(range(1, 11)))) == ["10 questions are held out, not 20"]


def test_an_unsplit_set_is_not_yet_a_problem() -> None:
    assert split_problems(DRAFT) == []


def test_a_partly_split_set_is_reported() -> None:
    questions = _split(set(range(1, FINAL_SIZE + 1)))
    questions[-1]["split"] = None
    assert "1 questions have no split" in split_problems(questions)


def test_a_line_that_is_not_json_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text('{"qid": "g001"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_gold(path)
