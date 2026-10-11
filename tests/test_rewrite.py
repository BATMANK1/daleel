"""Tests for rewriting questions into search queries, and for measuring it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from daleel.answer.llm import GENERATOR, Reply, Request
from daleel.answer.rewrite import SYSTEM_REWRITE, queries, rewrite_request, rewriter
from daleel.retrieve.dense import DOCUMENT, QUESTION, VectorStore
from daleel.retrieve.rerank import ScoreStore


def test_the_rewrite_request_asks_for_queries() -> None:
    asked = rewrite_request("كم الرسوم؟")
    assert asked.model == GENERATOR and asked.system == SYSTEM_REWRITE
    assert asked.prompt == "Question: كم الرسوم؟" and asked.purpose == "rewrite"


def test_queries_lose_list_markers_and_quotes_but_keep_their_numbers() -> None:
    reply = '1. رسوم الوحدة\n- رسوم الساعة\n\n• "الغياب"\n3.75 المعدل\nرسوم الوحدة'
    assert queries(reply) == ["رسوم الوحدة", "رسوم الساعة", "الغياب"]
    assert queries("3.75 المعدل التراكمي") == ["3.75 المعدل التراكمي"]
    assert queries("") == []


def test_the_rewriter_searches_the_question_first_and_once() -> None:
    asked: list[Request] = []

    def ask(request: Request) -> Reply:
        asked.append(request)
        return Reply("كم الرسوم؟\nرسوم الوحدة الدراسية", "STOP", "v1", 10, 5, 0, 0.4)

    assert rewriter(ask)("كم الرسوم؟") == ["كم الرسوم؟", "رسوم الوحدة الدراسية"]
    assert [request.purpose for request in asked] == ["rewrite"]


# --- scripts/eval_rewrite.py --------------------------------------------------------

GUIDE = "student_guide_2025"
EVIDENCE = "المبلغ 250 ريال عند القبول"
CHUNKS = [
    {"chunk_id": f"c{page}", "doc_id": GUIDE, "page": page, "text": text}
    for page, text in enumerate(
        [
            "رسوم الكتب المستعارة",
            EVIDENCE,
            "الغياب عشرون بالمئة",
            "مواعيد الاختبارات النهائية",
            "شروط التحويل بين التخصصات",
            "المكافأة الشهرية للطلاب",
            "قواعد السكن الطلابي",
        ],
        start=1,
    )
]
ASKED = "كم أدفع؟"
QUERY = "رسوم القبول"


def question(qid: str, text: str, **fields: object) -> dict:
    return {
        "qid": qid,
        "type": "single_clause",
        "lang": "ar",
        "question": text,
        "answerability_status": fields.get("status", "supported"),
        "split": fields.get("split", "dev"),
        "source_refs": [
            {"doc_id": GUIDE, "pdf_page": 2, "role": "answer_evidence", "quotes": ["المبلغ 250"]}
        ],
    }


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def test_the_measure_compares_the_question_with_its_rewrites_on_dev_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    gold = [
        question("g1", ASKED),
        question("g2", "كم تدفع أرامكو؟", status="outside_corpus"),
        question("g3", "متى أدفع؟", split="final"),
    ]
    texts = [chunk["text"] for chunk in CHUNKS]
    store = VectorStore()
    store.add(DOCUMENT, texts, [unit(1, place) for place in range(len(texts))])
    store.add(QUESTION, [q["question"] for q in gold], [unit(1, 0)] * 3)
    store.save(tmp_path / "dense" / "bge-m3.npz")
    scores = ScoreStore()
    # As asked, the evidence is reranked last of seven.
    scores.add(ASKED, texts, [-5.0 if text == EVIDENCE else 1.0 for text in texts])
    scores.save(tmp_path / "rerank.json")

    class StandIn:
        """The encoder and the reranker, for the rewritten query alone."""

        def encode(self, texts: list[str], **options: object) -> np.ndarray:
            assert texts == [QUERY]
            return np.array([unit(1, 0)])

        def predict(self, pairs: list, **options: object) -> list[float]:
            assert {asked for asked, _ in pairs} == {QUERY}
            return [9.0 if text == EVIDENCE else 0.0 for _, text in pairs]

    class FakeClient:
        asked = 0

        def ask(self, request: Request) -> Reply:
            FakeClient.asked += 1
            assert request.purpose == "rewrite" and request.prompt == f"Question: {ASKED}"
            return Reply(f"1. {QUERY}", "STOP", "v1", 80, 6, 0, 0.7)

    eval_rewrite = script("eval_rewrite")
    monkeypatch.setattr(eval_rewrite, "load_model", lambda *args: StandIn())
    monkeypatch.setattr(eval_rewrite, "load_reranker", lambda *args: StandIn())
    monkeypatch.setattr(eval_rewrite, "Client", FakeClient)
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", CHUNKS))]
    argv += ["--gold", str(write_lines(tmp_path / "gold.jsonl", gold))]
    argv += ["--dense", str(tmp_path / "dense"), "--rerank", str(tmp_path / "rerank.json")]
    argv += ["--out", str(tmp_path / "runs")]
    assert eval_rewrite.main(argv) == 0
    out = capsys.readouterr().out
    assert "| as asked | 0.000 | 1.000 | 0.000 | 0.143 | 0 | 0 |" in out
    assert "| queries alone | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 |" in out
    assert "| as asked and queries, fused |" in out
    # Only the answerable dev question is rewritten, once.
    assert FakeClient.asked == 1
    [saved] = [json.loads(line) for line in (tmp_path / "runs" / "rewrite-dev.jsonl").open()]
    assert saved["qid"] == "g1" and saved["queries"] == [QUERY]
