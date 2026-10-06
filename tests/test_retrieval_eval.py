"""Tests for measuring retrieval against the gold set's evidence."""

from __future__ import annotations

import pytest

from daleel.eval.retrieval import Ranked, by_page, evaluate, evidence, rank_evidence, summary

GUIDE, CHARTER = "student_guide_2025", "student_charter"


def chunk(chunk_id: str, doc_id: str, page: int, text: str) -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "page": page, "text": text}


CHUNKS = [
    chunk("g30a", GUIDE, 30, "يمنح الطالب تقدير (ح) في حال تجاوز نسبة الغياب (20%)"),
    chunk("g30b", GUIDE, 30, "الفئة: الطلبة، عدد الكتب: 5 كتب"),
    chunk("g30c", GUIDE, 30, "مدة الاعارة: 14 يوم"),
    chunk("g31a", GUIDE, 31, "تجاوز نسبة الغياب (20%) في صفحة أخرى"),
    chunk("ch4a", CHARTER, 4, "يحق للطالب المحافظة على خصوصية بياناته"),
]


def ref(doc_id: str, page: int, quotes: list, role: str = "answer_evidence") -> dict:
    return {"doc_id": doc_id, "pdf_page": page, "role": role, "quotes": quotes}


def question(qid: str, refs: list, status: str = "supported", **fields: str) -> dict:
    return {
        "qid": qid,
        "type": fields.get("type", "single_clause"),
        "lang": fields.get("lang", "ar"),
        "question": fields.get("question", qid),
        "answerability_status": status,
        "source_refs": refs,
    }


def test_a_quote_is_held_by_the_chunks_of_its_page_that_hold_it() -> None:
    found = evidence(question("g1", [ref(GUIDE, 30, ["نسبة الغياب (20%)"])]), CHUNKS)
    assert found == [frozenset({"g30a"})]


def test_a_quote_s_parts_must_all_be_in_one_chunk() -> None:
    found = evidence(
        question("g1", [ref(GUIDE, 30, [["الطلبة", "5 كتب"], ["الطلبة", "14 يوم"]])]), CHUNKS
    )
    assert found == [frozenset({"g30b"}), frozenset()]


def test_a_quote_is_found_after_normalization() -> None:
    found = evidence(question("g1", [ref(CHARTER, 4, ["خصوصيه بياناته"])]), CHUNKS)
    assert found == [frozenset({"ch4a"})]


def test_chunks_arranged_by_page_once_give_the_same_evidence() -> None:
    q = question("g1", [ref(GUIDE, 30, ["نسبة الغياب"]), ref(CHARTER, 4, ["خصوصية"])])
    assert evidence(q, by_page(CHUNKS)) == evidence(q, CHUNKS)


def test_only_the_pages_that_answer_count() -> None:
    refs = [ref(GUIDE, 31, ["نسبة الغياب"], role="related_only"), ref(CHARTER, 4, ["خصوصية"])]
    assert evidence(question("g1", refs), CHUNKS) == [frozenset({"ch4a"})]


def test_each_quote_is_ranked_where_its_first_holder_is() -> None:
    q = question("g1", [ref(GUIDE, 30, ["نسبة الغياب"]), ref(CHARTER, 4, ["خصوصية"])])
    ranked = rank_evidence(q, evidence(q, CHUNKS), ["g31a", "x", "g30a", "ch4a"])
    assert ranked.ranks == (3, 4)
    assert ranked.recall(1) == 0 and ranked.recall(3) == 0.5 and ranked.recall(4) == 1
    assert not ranked.complete(3) and ranked.complete(4)
    assert ranked.reciprocal_rank() == pytest.approx(1 / 3)
    assert ranked.reciprocal_rank(2) == 0


def test_a_quote_no_chunk_holds_is_counted_apart() -> None:
    q = question("g1", [ref(GUIDE, 30, ["لا يوجد هذا النص"])])
    ranked = rank_evidence(q, evidence(q, CHUNKS), ["g30a"])
    assert (ranked.ranks, ranked.unheld) == ((None,), 1)


def test_only_answerable_questions_are_measured() -> None:
    questions = [
        question("g1", [ref(GUIDE, 30, ["نسبة الغياب"])], question="الغياب"),
        question("g2", [], status="outside_corpus"),
        question("g3", [ref(GUIDE, 30, ["x"], role="related_only")], status="unsupported_detail"),
    ]
    asked = []

    def search(text: str, k: int) -> list[str]:
        asked.append((text, k))
        return ["g30a"]

    results = evaluate(questions, CHUNKS, search)
    assert [result.qid for result in results] == ["g1"]
    assert asked == [("الغياب", 20)]


def test_the_summary_averages_over_questions() -> None:
    results = [
        Ranked("g1", "numeric", "ar", (1, 7), 0),
        Ranked("g2", "numeric", "ar", (None,), 1),
    ]
    found = summary(results, (5, 10))
    assert found == {
        "questions": 2,
        "held": pytest.approx(2 / 3),
        "recall@5": 0.25,
        "recall@10": 0.5,
        "complete@5": 0,
        "complete@10": 0.5,
        "mrr@10": 0.5,
    }
    assert summary([]) == {}
