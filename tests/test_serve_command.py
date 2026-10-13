"""Tests for daleel serve, with stand-ins for the two models, the model client and uvicorn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from daleel import cli
from daleel.answer.llm import Reply, Request
from daleel.chunk.chunker import write_chunks
from daleel.retrieve.dense import DOCUMENT, VectorStore, chunk_texts
from daleel.retrieve.hybrid import chosen_retrieval
from daleel.serve import load

FEES = "رسوم الوحدة الدراسية 250 ريالاً"
CHUNKS = [
    {
        "chunk_id": chunk_id,
        "doc_id": "student_guide_2025",
        "doc_title": "دليل الطالب",
        "page": page,
        "section_heading": "",
        "clause_no": None,
        "text": text,
    }
    for chunk_id, page, text in [
        ("absence", 30, "يحرم الطالب إذا تجاوز غيابه 20%"),
        ("library", 8, "يغرم المستعير ريالاً عن كل يوم تأخير"),
        ("fees", 28, FEES),
    ]
]
ASKED = "كم رسوم الوحدة؟"
OUTSIDE = "كم تدفع أرامكو لطالب التدريب؟"


class Encoder:
    """bge-m3's stand-in: every question points at the first chunk, absence."""

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class Reranker:
    """The cross-encoder's stand-in: the fee chunk matches a question about fees,
    and nothing matches a question about Aramco."""

    def predict(self, pairs: list[tuple[str, str]], **options: Any) -> list[float]:
        return [
            -9.0 if "أرامكو" in question else 5.0 if text == FEES else 0.0
            for question, text in pairs
        ]


class FakeClient:
    """The model client's stand-in, kept per folder of stored replies."""

    made: ClassVar[list[FakeClient]] = []

    def __init__(self, root: Path) -> None:
        self.root = root
        self.asked: list[Request] = []
        FakeClient.made.append(self)

    def ask(self, request: Request) -> Reply:
        self.asked.append(request)
        return Reply("رسوم الوحدة 250 ريالاً [1].", "STOP", "v1", 100, 20, 0, 0.5)


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The files daleel serve reads, its models replaced, and what it passed to uvicorn."""
    chunks = write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    store = VectorStore()
    store.add(DOCUMENT, chunk_texts(CHUNKS), np.eye(3, dtype=np.float32))
    store.save(tmp_path / "dense" / "bge-m3.npz")
    monkeypatch.setattr(load, "load_model", lambda *args: Encoder())
    monkeypatch.setattr(load, "load_reranker", lambda *args: Reranker())
    monkeypatch.setattr(load, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    FakeClient.made = []
    ran: dict[str, Any] = {}

    def run(app: FastAPI, **options: Any) -> None:
        ran.update(options, app=app)

    monkeypatch.setattr("uvicorn.run", run)
    argv = ["serve", "--chunks", str(chunks), "--dense", str(tmp_path / "dense")]
    argv += ["--log", str(tmp_path / "serve" / "queries.jsonl")]
    argv += ["--replies", str(tmp_path / "serve" / "llm")]
    return {"argv": argv, "ran": ran, "tmp": tmp_path}


def answer_of(response_text: str) -> dict:
    for block in response_text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        if fields["event"] == "answer":
            return json.loads(fields["data"])
    raise AssertionError("no answer event")


def test_serve_listens_on_this_machine_alone_and_writes_no_access_log(served: dict) -> None:
    assert cli.main(served["argv"]) == 0
    ran = served["ran"]
    assert (ran["host"], ran["port"]) == ("127.0.0.1", 8000)
    assert ran["access_log"] is False
    assert isinstance(ran["app"], FastAPI)
    assert FakeClient.made[0].root == served["tmp"] / "serve" / "llm"


def test_the_served_app_reranks_live_and_answers_with_the_flow(served: dict) -> None:
    cli.main(served["argv"])
    client = TestClient(served["ran"]["app"])
    assert client.get("/health").json() == {"status": "ok", "chunks": 3}
    # bge-m3's stand-in puts absence first; the reranker puts fees first.
    found = client.post("/search", json={"query": ASKED, "k": 3}).json()["results"]
    assert [result["chunk_id"] for result in found] == ["fees", "absence", "library"]
    answer = answer_of(client.post("/ask", json={"question": ASKED}).text)
    assert answer["outcome"] == "answered" and answer["sources"][0]["chunk_id"] == "fees"
    [line] = (served["tmp"] / "serve" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["question"] == ASKED


def test_a_question_nothing_matches_is_refused_without_asking_the_model(served: dict) -> None:
    cli.main(served["argv"])
    client = TestClient(served["ran"]["app"])
    answer = answer_of(client.post("/ask", json={"question": OUTSIDE}).text)
    assert (answer["outcome"], answer["reason"]) == ("refused", "low confidence")
    assert FakeClient.made[0].asked == []


def test_serve_stops_before_loading_anything_when_a_file_or_the_key_is_missing(
    served: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = served["argv"]
    no_chunks = [*argv[:2], str(tmp_path / "none.jsonl"), *argv[3:]]
    assert cli.main(no_chunks) == 2
    assert "daleel chunk" in capsys.readouterr().err
    no_vectors = [*argv[:4], str(tmp_path / "empty"), *argv[5:]]
    assert cli.main(no_vectors) == 2
    assert "encode_dense.py" in capsys.readouterr().err
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.chdir(tmp_path)
    assert cli.main(argv) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert served["ran"] == {} and FakeClient.made == []


def test_live_reranking_needs_a_score_function(tmp_path: Path) -> None:
    store = VectorStore()
    store.add(DOCUMENT, chunk_texts(CHUNKS), np.eye(3, dtype=np.float32))
    store.save(tmp_path / "bge-m3.npz")
    with pytest.raises(ValueError, match="score function"):
        chosen_retrieval(CHUNKS, tmp_path, None)
