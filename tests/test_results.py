"""Tests for the results tables built from judged runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

GUIDE = "student_guide_2025"
CHUNKS = [
    {"chunk_id": "fee", "doc_id": GUIDE, "page": 28, "text": "رسوم الوحدة 250 ريال"},
    {"chunk_id": "gpa", "doc_id": GUIDE, "page": 40, "text": "المعدل 3.75"},
]
GOLD = [
    {
        "qid": "g1",
        "type": "numeric",
        "question": "كم رسوم الوحدة؟",
        "answerability_status": "supported",
        "answer_numeric": 250,
        "source_refs": [
            {"doc_id": GUIDE, "pdf_page": 28, "role": "answer_evidence", "quotes": ["250"]}
        ],
    },
    {
        "qid": "g2",
        "type": "single_clause",
        "question": "ما المعدل؟",
        "answerability_status": "supported",
        "source_refs": [
            {"doc_id": GUIDE, "pdf_page": 40, "role": "answer_evidence", "quotes": ["3.75"]}
        ],
    },
    {
        "qid": "g3",
        "type": "out_of_scope",
        "question": "كم تدفع أرامكو؟",
        "answerability_status": "outside_corpus",
        "source_refs": [],
    },
]
BEHAVIOR = {"g1": "answer", "g2": "answer", "g3": "refuse"}


def record(qid: str, answer: str, judge: str, sources: list[str], **fields: object) -> dict:
    return {
        "qid": qid,
        "type": next(q["type"] for q in GOLD if q["qid"] == qid),
        "behavior": BEHAVIOR[qid],
        "answer": answer,
        "sources": sources,
        "judge": judge,
        "seconds": fields.get("seconds", 1.0),
        "tokens_in": fields.get("tokens_in", 100),
        "tokens_out": 10,
        "usd_at_paid_rates": fields.get("usd", 0.001),
    }


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def test_the_tables_count_the_judge_s_labels_and_work_out_the_measures_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    # The flow: right with citations, right but citing the wrong chunk, refusal.
    flow = [
        record("g1", "الرسوم 250 [1].", "correct", ["fee", "gpa"], seconds=2.0),
        record("g2", "المعدل 3.75 [1].", "partial", ["fee", "gpa"], seconds=4.0),
        record("g3", "لا تجيب الوثائق المتاحة عن هذا السؤال.", "correct", ["fee"]),
    ]
    # Arm A: no sources; it invents the outside-corpus amount.
    plain = [
        record("g1", "لا أعرف إجابة موثوقة عن هذا السؤال.", "incorrect", [], usd=None),
        record("g2", "المعدل 3.75.", "correct", [], usd=None),
        record("g3", "تدفع أرامكو 3000 ريال.", "hallucinated", [], usd=None),
    ]
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", CHUNKS))]
    argv += ["--gold", str(write_lines(tmp_path / "gold.jsonl", GOLD))]
    runs = [write_lines(tmp_path / "A-dev-judged.jsonl", plain)]
    runs.append(write_lines(tmp_path / "flow-dev-judged.jsonl", flow))
    assert script("results").main([*argv, "--runs", *map(str, runs)]) == 0
    out = capsys.readouterr().out
    # n, accuracy, interval, correct or partial, numbers, precision, hallucinated,
    # hallucinated outside the corpus, accuracy minus the flow's.
    a_row = next(line for line in out.splitlines() if line.startswith("| A-dev | 3 | 0.33 |"))
    assert "| 1/3 | 0/1 | - | 1/3 | 1/1 | -0.33 (" in a_row
    assert "| flow-dev | 3 | 0.67 |" in out and "| 3/3 | 1/1 | 0.50 | 0/3 | 0/1 | - |" in out
    assert "| flow-dev | 2.0 | 4.0 | 100 | 10 | 1.00 |" in out
    assert "| A-dev | 1.0 | 1.0 | 100 | 10 | - |" in out
    assert "T8, flow-dev |" in out and "| numeric | 1 | 1 | 0 | 0 | 0 | 1.00 |" in out


def test_a_run_not_yet_judged_or_an_unknown_baseline_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", CHUNKS))]
    argv += ["--gold", str(write_lines(tmp_path / "gold.jsonl", GOLD))]
    unjudged = {k: v for k, v in record("g1", "250 [1]", "correct", ["fee"]).items()}
    del unjudged["judge"]
    run = write_lines(tmp_path / "C-dev.jsonl", [unjudged])
    results = script("results")
    assert results.main([*argv, "--runs", str(run)]) == 2
    assert "is not judged" in capsys.readouterr().err
    judged = write_lines(tmp_path / "C-dev-judged.jsonl", [record("g1", "250", "correct", [])])
    assert results.main([*argv, "--runs", str(judged), "--against", "B-dev"]) == 2
    assert "no run named B-dev" in capsys.readouterr().err


def test_a_run_whose_answers_cite_the_calendar_s_weeks_is_measured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    # Two semesters' first days: the first semester runs until the second starts.
    cards = [
        {
            "chunk_id": f"academic_weeks_1448_p{page}_c1",
            "doc_id": "academic_weeks_1448",
            "page": page,
            "section_heading": name,
            "text": f"{name}، بداية الفصل الدراسي: الأحد (Sun) {day}م",
        }
        for page, name, day in [
            (1, "الفصل الدراسي الأول (481)", "2026/08/23"),
            (2, "الفصل الدراسي الثاني (482)", "2027/01/17"),
        ]
    ]
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", [*CHUNKS, *cards]))]
    argv += ["--gold", str(write_lines(tmp_path / "gold.jsonl", GOLD))]
    flow = [record("g1", "الرسوم 250 [1] في الأسبوع 8 [2].", "correct", ["fee", "calendar_weeks"])]
    run = write_lines(tmp_path / "flow-dev-judged.jsonl", flow)
    assert script("results").main([*argv, "--runs", str(run)]) == 0
    # The computed source holds no gold quote, so it counts against precision.
    assert "| flow-dev | 1 | 1.00 |" in capsys.readouterr().out
