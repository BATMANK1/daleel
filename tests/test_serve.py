"""Tests for the web service, with a fake search and a fake model."""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from daleel.answer.arms import DECLINE
from daleel.answer.llm import LlmError, Reply, Request
from daleel.serve.app import UNAVAILABLE, create_app
from daleel.serve.privacy import REDACTED

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
# Every key a line of the query log has, and no other.
LOG_FIELDS = {
    "at",
    "question",
    "language",
    "outcome",
    "reason",
    "attempts",
    "confidence",
    "sources",
    "cited",
    "answer",
    "seconds",
    "tokens_in",
    "tokens_out",
}


class Search:
    """Ranks the chunks c1, c2, ... for every query, and keeps what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> list[str]:
        self.calls.append((query, k))
        return RANKING[:k]


class Model:
    """Answers with the given texts in turn, and keeps what it was asked."""

    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.asked: list[Request] = []

    def __call__(self, asked: Request) -> Reply:
        self.asked.append(asked)
        return Reply(self.texts[len(self.asked) - 1], "STOP", "v1", 100, 20, 0, 0.5)


def ticking(step: float = 2.5) -> Callable[[], float]:
    """A clock that moves on by `step` seconds each time it is read."""
    ticks = itertools.count(start=0.0, step=step)
    return lambda: next(ticks)


def service(
    model: Callable[[Request], Reply], search: Search | None = None, **options: Any
) -> TestClient:
    app = create_app(search or Search(), CHUNKS, model, clock=ticking(), **options)
    return TestClient(app)


def events(text: str) -> list[tuple[str, dict]]:
    """The server-sent events of a response: each one's name and its data."""
    found = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        found.append((fields["event"], json.loads(fields["data"])))
    return found


def steps(found: Sequence[tuple[str, dict]]) -> list[str]:
    return [data["step"] for name, data in found if name == "step"]


def logged(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_health_says_how_many_chunks_are_served() -> None:
    response = service(Model()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 12}


def test_search_gives_the_ranked_chunks_and_where_they_come_from() -> None:
    search = Search()
    client = service(Model(), search)
    response = client.post("/search", json={"query": "رسوم الوحدة", "k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert results[1] == {
        "rank": 2,
        "chunk_id": "c2",
        "doc_id": "student_guide_2025",
        "doc_title": "دليل الطالب",
        "page": 2,
        "section_heading": "الرسوم الدراسية",
        "text": "نص المقطع 2",
    }
    assert client.post("/search", json={"query": "رسوم الوحدة"}).json()["results"][4]["rank"] == 5
    assert search.calls == [("رسوم الوحدة", 3), ("رسوم الوحدة", 5)]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/ask", {"question": "   "}),
        ("/ask", {"question": "س" * 501}),
        ("/ask", {}),
        ("/search", {"query": ""}),
        ("/search", {"query": "رسوم", "k": 21}),
        ("/search", {"query": "رسوم", "k": 0}),
    ],
)
def test_a_question_empty_or_too_long_is_rejected_before_any_work(
    path: str, body: dict, tmp_path: Path
) -> None:
    search, model = Search(), Model()
    response = service(model, search, log=tmp_path / "queries.jsonl").post(path, json=body)
    assert response.status_code == 422
    assert search.calls == [] and model.asked == []
    assert not (tmp_path / "queries.jsonl").exists()


def test_an_answer_streams_its_steps_then_the_answer_with_the_sources_it_cites() -> None:
    response = service(Model("رسوم الوحدة 250 ريالاً [2].")).post(
        "/ask", json={"question": QUESTION}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    found = events(response.text)
    assert steps(found) == ["rewrite", "retrieve", "generate", "verify"]
    name, answer = found[-1]
    assert name == "answer"
    assert answer == {
        "answer": "رسوم الوحدة 250 ريالاً [2].",
        "outcome": "answered",
        "reason": "",
        "language": "ar",
        # Only the sources the answer cites, under the numbers it cites them by.
        "sources": [
            {
                "n": 2,
                "chunk_id": "c2",
                "doc_id": "student_guide_2025",
                "doc_title": "دليل الطالب",
                "page": 2,
                "section_heading": "الرسوم الدراسية",
                "text": "نص المقطع 2",
            }
        ],
    }


def test_a_refusal_streams_the_fixed_sentence_and_no_sources() -> None:
    response = service(Model("250.", "250 ريالاً.")).post("/ask", json={"question": QUESTION})
    found = events(response.text)
    assert steps(found)[-1] == "refuse"
    assert found[-1] == (
        "answer",
        {
            "answer": DECLINE["ar"],
            "outcome": "refused",
            "reason": "no citation",
            "language": "ar",
            "sources": [],
        },
    )


def test_a_failing_model_ends_the_stream_with_an_error_and_is_logged(tmp_path: Path) -> None:
    def broken(asked: Request) -> Reply:
        raise LlmError("gemini-3.5-flash-lite: HTTP 500: internal")

    log = tmp_path / "queries.jsonl"
    response = service(broken, log=log).post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    found = events(response.text)
    assert steps(found) == ["rewrite", "retrieve"]
    assert found[-1] == ("error", {"message": UNAVAILABLE["ar"]})
    assert "answer" not in [name for name, _ in found]
    [line] = logged(log)
    assert line["outcome"] == "error" and line["reason"] == "LlmError"
    assert line["answer"] == "" and line["cited"] == []


def test_the_question_is_redacted_before_search_the_model_or_the_log_see_it(
    tmp_path: Path,
) -> None:
    search, model = Search(), Model("رسوم الوحدة 250 ريالاً [1].")
    log = tmp_path / "queries.jsonl"
    asked = "رقمي الجامعي 441234567، كم رسوم الوحدة؟"
    service(model, search, log=log).post("/ask", json={"question": asked})
    assert search.calls == [(f"رقمي الجامعي {REDACTED}، كم رسوم الوحدة؟", 5)]
    assert "441234567" not in model.asked[0].prompt and REDACTED in model.asked[0].prompt
    [line] = logged(log)
    assert line["question"] == f"رقمي الجامعي {REDACTED}، كم رسوم الوحدة؟"


def test_the_query_log_keeps_what_was_asked_and_answered_and_nothing_about_who(
    tmp_path: Path,
) -> None:
    log = tmp_path / "logs" / "queries.jsonl"
    client = service(
        Model("رسوم الوحدة 250 ريالاً [2]."), log=log, confidence=lambda *_: 1.5, threshold=-4.0
    )
    headers = {
        "User-Agent": "Mozilla/5.0 agent-of-student",
        "X-Forwarded-For": "203.0.113.7",
        "Cookie": "session=abc123",
    }
    client.post("/ask", json={"question": QUESTION}, headers=headers)
    [line] = logged(log)
    assert set(line) == LOG_FIELDS
    # The hour alone, so that a line cannot be matched to a moment.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00Z", line["at"])
    assert line["question"] == QUESTION and line["language"] == "ar"
    assert (line["outcome"], line["reason"], line["attempts"]) == ("answered", "", 1)
    assert line["confidence"] == 1.5
    assert line["sources"] == RANKING[:5] and line["cited"] == ["c2"]
    assert line["answer"] == "رسوم الوحدة 250 ريالاً [2]."
    assert (line["seconds"], line["tokens_in"], line["tokens_out"]) == (2.5, 100, 20)
    text = log.read_text(encoding="utf-8")
    for trace in ("203.0.113.7", "agent-of-student", "abc123", "testclient"):
        assert trace not in text


def test_searches_are_redacted_and_not_logged(tmp_path: Path) -> None:
    search, log = Search(), tmp_path / "queries.jsonl"
    service(Model(), search, log=log).post("/search", json={"query": "رسوم 0551234567"})
    assert search.calls == [(f"رسوم {REDACTED}", 5)]
    assert not log.exists()


def test_metrics_count_asks_by_outcome_with_their_time_and_tokens() -> None:
    client = service(Model("رسوم الوحدة 250 ريالاً [1].", "250.", "250 ريالاً."))
    client.post("/ask", json={"question": QUESTION})
    client.post("/ask", json={"question": "كم رسوم السكن؟"})
    client.post("/search", json={"query": "رسوم"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'daleel_asks_total{outcome="answered"} 1.0' in text
    assert 'daleel_asks_total{outcome="refused"} 1.0' in text
    assert "daleel_ask_seconds_count 2.0" in text and "daleel_ask_seconds_sum 5.0" in text
    # Three replies: one for the first question, two for the second.
    assert 'daleel_model_tokens_total{direction="in"} 300.0' in text
    assert 'daleel_model_tokens_total{direction="out"} 60.0' in text
    assert "daleel_searches_total 1.0" in text


def test_two_services_in_one_process_count_apart() -> None:
    first = service(Model("رسوم الوحدة 250 ريالاً [1]."))
    second = service(Model())
    first.post("/ask", json={"question": QUESTION})
    assert 'daleel_asks_total{outcome="answered"} 1.0' in first.get("/metrics").text
    assert 'daleel_asks_total{outcome="answered"}' not in second.get("/metrics").text


def test_questions_reach_the_model_one_at_a_time() -> None:
    lock = threading.Lock()
    busy = {"now": 0, "most": 0}

    def slow(asked: Request) -> Reply:
        with lock:
            busy["now"] += 1
            busy["most"] = max(busy["most"], busy["now"])
        time.sleep(0.2)
        with lock:
            busy["now"] -= 1
        return Reply("رسوم الوحدة 250 ريالاً [1].", "STOP", "v1", 100, 20, 0, 0.2)

    client = service(slow)
    asks = [
        threading.Thread(target=client.post, args=("/ask",), kwargs={"json": {"question": q}})
        for q in (QUESTION, "كم رسوم السكن؟", "كم رسوم الكتب؟")
    ]
    for ask in asks:
        ask.start()
    for ask in asks:
        ask.join()
    assert busy["most"] == 1


def test_the_page_is_arabic_right_to_left_and_says_what_daleel_is() -> None:
    response = service(Model()).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<html lang="ar" dir="rtl">' in response.text
    assert "أداة مساعدة" in response.text
    # Nothing is loaded from another site, which would learn who is asking.
    assert "://" not in response.text


def test_the_page_s_fonts_are_served_from_this_server() -> None:
    client = service(Model())
    fonts = re.findall(r'url\("(/static/fonts/[^"]+)"\)', client.get("/").text)
    assert len(fonts) >= 4
    for font in fonts:
        response = client.get(font)
        assert response.status_code == 200, font
        assert response.headers["content-type"] == "font/woff2"


def sources_of(found: Sequence[tuple[str, dict]]) -> list[Mapping[str, Any]]:
    return next(data for name, data in found if name == "answer")["sources"]


def test_an_english_question_is_answered_with_its_cited_sources_in_order() -> None:
    response = service(Model("The fee is 250 riyals [3][1].")).post(
        "/ask", json={"question": "How much is the fee per credit hour?"}
    )
    found = events(response.text)
    assert found[-1][1]["language"] == "en"
    assert [source["n"] for source in sources_of(found)] == [3, 1]


CODE = "pilot-code-0123"


def test_with_an_access_code_only_requests_that_send_it_are_answered(tmp_path: Path) -> None:
    model = Model("رسوم الوحدة 250 ريالاً [1].")
    search = Search()
    log = tmp_path / "queries.jsonl"
    client = service(model, search, log=log, code=CODE)
    asked = {"question": QUESTION}
    for headers in ({}, {"X-Access-Code": "pilot-code-0124"}, {"X-Access-Code": ""}):
        assert client.post("/ask", json=asked, headers=headers).status_code == 401
        assert client.post("/search", json={"query": "رسوم"}, headers=headers).status_code == 401
        assert client.get("/metrics", headers=headers).status_code == 401
    # Turned away before the flow: nothing searched, asked or logged.
    assert search.calls == [] and model.asked == [] and not log.exists()

    allowed = {"X-Access-Code": CODE}
    response = client.post("/ask", json=asked, headers=allowed)
    assert response.status_code == 200 and events(response.text)[-1][0] == "answer"
    assert client.post("/search", json={"query": "رسوم"}, headers=allowed).status_code == 200
    assert client.get("/metrics", headers=allowed).status_code == 200
    # The page and the health check need no code: they hold nothing from the documents.
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_without_an_access_code_no_request_needs_one() -> None:
    client = service(Model("رسوم الوحدة 250 ريالاً [1]."))
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200
    assert client.get("/metrics").status_code == 200


def test_the_service_publishes_no_generated_api_pages() -> None:
    client = service(Model(), code=CODE)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_the_page_sends_the_code_its_invitation_link_carries() -> None:
    page = service(Model()).get("/").text
    # The code comes after #, which the browser never sends to the server.
    assert "location.hash" in page
    # Also when the code is added to the address of a page already open.
    assert 'addEventListener("hashchange"' in page
    assert '"X-Access-Code"' in page
    assert "response.status === 401" in page
