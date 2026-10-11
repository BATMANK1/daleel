"""Tests for the answering flow, with a fake search and a fake model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from daleel.answer.arms import DECLINE, SYSTEM_C
from daleel.answer.flow import build_flow, language
from daleel.answer.llm import Reply, Request

QUESTION = "كم رسوم الوحدة الدراسية؟"
CHUNKS = [
    {
        "chunk_id": f"c{n}",
        "doc_id": "student_guide_2025",
        "doc_title": "دليل الطالب",
        "page": n,
        "section_heading": "الرسوم الدراسية",
        "clause_no": None,
        "text": f"نص المقطع {n}",
    }
    for n in range(1, 13)
]
RANKING = [chunk["chunk_id"] for chunk in CHUNKS]


class Search:
    """Ranks the chunks c1, c2, ... for every query, unless given a ranking for it."""

    def __init__(self, rankings: Mapping[str, list[str]] | None = None) -> None:
        self.rankings = rankings or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> list[str]:
        self.calls.append((query, k))
        return self.rankings.get(query, RANKING)[:k]


class Model:
    """Answers with the given texts in turn, and keeps what it was asked."""

    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.asked: list[Request] = []

    def __call__(self, asked: Request) -> Reply:
        self.asked.append(asked)
        return Reply(self.texts[len(self.asked) - 1], "STOP", "v1", 100, 20, 0, 0.5)


def given(asked: Request) -> int:
    """How many sources a request gave the model."""
    return asked.prompt.count("<source n=")


def test_a_cited_answer_comes_back_from_one_request_with_five_sources() -> None:
    search, model = Search(), Model("رسوم الوحدة 250 ريالاً [1].")
    state = build_flow(search, CHUNKS, model).invoke({"question": QUESTION})
    assert state["outcome"] == "answered" and state["answer"] == "رسوم الوحدة 250 ريالاً [1]."
    assert state["cited"] == [1] and state["attempts"] == 1
    assert [source["chunk_id"] for source in state["sources"]] == RANKING[:5]
    assert search.calls == [(QUESTION, 5)]
    [asked] = model.asked
    assert asked.system == SYSTEM_C and asked.purpose == "arm C" and given(asked) == 5
    assert len(state["replies"]) == 1


def test_the_graph_has_the_designed_steps() -> None:
    flow = build_flow(Search(), CHUNKS, Model())
    assert {"rewrite", "retrieve", "generate", "verify", "refuse"} <= set(flow.get_graph().nodes)


def test_an_answer_without_citations_is_asked_again_with_ten_sources() -> None:
    search, model = Search(), Model("رسوم الوحدة 250 ريالاً.", "رسوم الوحدة 250 ريالاً [7].")
    state = build_flow(search, CHUNKS, model).invoke({"question": QUESTION})
    assert state["outcome"] == "answered" and state["cited"] == [7] and state["attempts"] == 2
    assert search.calls == [(QUESTION, 5), (QUESTION, 10)]
    assert [given(asked) for asked in model.asked] == [5, 10]
    assert len(state["replies"]) == 2


def test_a_citation_of_a_source_not_given_is_caught_before_the_checker_runs() -> None:
    checks: list[tuple[str, str, int]] = []

    def check(question: str, answer: str, sources: Sequence[Mapping[str, Any]]) -> bool:
        checks.append((question, answer, len(sources)))
        return True

    model = Model("رسوم الوحدة 250 ريالاً [6].", "رسوم الوحدة 250 ريالاً [6].")
    state = build_flow(Search(), CHUNKS, model, verify=check).invoke({"question": QUESTION})
    # [6] names no source among the first five, and the sixth among ten.
    assert state["outcome"] == "answered" and state["cited"] == [6]
    assert checks == [(QUESTION, "رسوم الوحدة 250 ريالاً [6].", 10)]


def test_an_answer_the_checker_rejects_is_asked_again() -> None:
    verdicts = iter([False, True])
    model = Model("رسوم الوحدة 300 ريال [1].", "رسوم الوحدة 250 ريالاً [1].")
    flow = build_flow(Search(), CHUNKS, model, verify=lambda *_: next(verdicts))
    state = flow.invoke({"question": QUESTION})
    assert state["outcome"] == "answered" and state["answer"] == "رسوم الوحدة 250 ريالاً [1]."
    assert state["attempts"] == 2


@pytest.mark.parametrize(
    ("question", "lang"), [(QUESTION, "ar"), ("How much is the fee per credit hour?", "en")]
)
def test_a_question_never_answered_with_citations_is_refused_in_its_language(
    question: str, lang: str
) -> None:
    model = Model("250.", "250 per credit hour.")
    state = build_flow(Search(), CHUNKS, model).invoke({"question": question})
    assert state["outcome"] == "refused" and state["reason"] == "no citation"
    assert state["answer"] == DECLINE[lang] and state["cited"] == []
    assert len(model.asked) == 2


def test_a_decline_gets_a_second_look_with_wider_retrieval() -> None:
    model = Model(DECLINE["ar"], "تقبل الأعذار الطبية خلال أسبوع [8].")
    state = build_flow(Search(), CHUNKS, model).invoke({"question": QUESTION})
    assert state["outcome"] == "answered" and state["cited"] == [8]
    assert [given(asked) for asked in model.asked] == [5, 10]


def test_a_question_the_documents_do_not_answer_ends_declined() -> None:
    # The first decline names only what is missing and cites nothing.
    model = Model("لا تذكر الوثائق المتاحة مكافأة أرامكو.", DECLINE["ar"])
    state = build_flow(Search(), CHUNKS, model).invoke({"question": "كم تدفع أرامكو؟"})
    assert state["outcome"] == "declined" and state["reason"] == "declined"
    assert state["answer"] == DECLINE["ar"] and state["cited"] == []
    assert len(model.asked) == 2


def test_a_weak_retrieval_is_refused_without_asking_the_model() -> None:
    scored: list[tuple[str, list[str]]] = []

    def weak(question: str, sources: Sequence[Mapping[str, Any]]) -> float:
        scored.append((question, [source["chunk_id"] for source in sources]))
        return -5.0

    model = Model()
    flow = build_flow(Search(), CHUNKS, model, confidence=weak, threshold=-3.0)
    state = flow.invoke({"question": QUESTION})
    assert state["outcome"] == "refused" and state["reason"] == "low confidence"
    assert state["answer"] == DECLINE["ar"] and state["confidence"] == -5.0
    assert scored == [(QUESTION, RANKING[:5])] and model.asked == []


def test_a_retrieval_at_the_threshold_is_answered_and_its_score_kept() -> None:
    model = Model("رسوم الوحدة 250 ريالاً [1].")
    flow = build_flow(Search(), CHUNKS, model, confidence=lambda *_: -3.0, threshold=-3.0)
    state = flow.invoke({"question": QUESTION})
    assert state["outcome"] == "answered" and state["confidence"] == -3.0


def test_a_threshold_needs_a_confidence_to_compare_with() -> None:
    with pytest.raises(ValueError, match="confidence"):
        build_flow(Search(), CHUNKS, Model(), threshold=-3.0)


def test_nothing_found_is_refused_without_scoring_it_or_asking_the_model() -> None:
    def best(question: str, sources: Sequence[Mapping[str, Any]]) -> float:
        # Like the reranker's best score, which has no value for no sources.
        return max(float(len(source["text"])) for source in sources)

    model = Model()
    flow = build_flow(Search({QUESTION: []}), CHUNKS, model, confidence=best, threshold=-3.0)
    state = flow.invoke({"question": QUESTION})
    assert state["outcome"] == "refused" and state["reason"] == "no sources"
    assert state["confidence"] is None and model.asked == []


def test_an_answer_that_fails_a_check_is_not_called_refused_before_its_retry() -> None:
    model = Model("رسوم الوحدة 250 ريالاً.", "رسوم الوحدة 250 ريالاً [1].")
    updates = build_flow(Search(), CHUNKS, model).stream(
        {"question": QUESTION}, stream_mode="updates"
    )
    checked = [update["verify"] for update in updates if "verify" in update]
    assert [(check["outcome"], check["reason"]) for check in checked] == [
        ("", "no citation"),
        ("answered", ""),
    ]


def test_rewritten_queries_are_each_searched_and_fused() -> None:
    search = Search(
        {
            "رسوم الوحدة": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "رسوم الساعة المعتمدة": ["c9", "c3", "c8", "c1", "c7", "c6"],
        }
    )
    model = Model("رسوم الوحدة 250 ريالاً [1].")
    flow = build_flow(
        search, CHUNKS, model, rewrite=lambda _: ["رسوم الوحدة", "رسوم الساعة المعتمدة"]
    )
    state = flow.invoke({"question": QUESTION})
    assert search.calls == [("رسوم الوحدة", 5), ("رسوم الساعة المعتمدة", 5)]
    # Reciprocal rank fusion of the two rankings, the first five.
    assert [source["chunk_id"] for source in state["sources"]] == ["c1", "c3", "c9", "c2", "c8"]
    # The model is asked the student's own question.
    assert model.asked[0].prompt.endswith(f"Question: {QUESTION}")


def test_a_question_with_any_arabic_letter_is_arabic() -> None:
    assert language(QUESTION) == "ar" and language("ما هو GPA؟") == "ar"
    assert language("What is the GPA?") == "en"
