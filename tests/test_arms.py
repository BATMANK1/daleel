"""Tests for the answering arms and the measures that need no judge."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from daleel.answer.arms import (
    DECLINE,
    GAP,
    SYSTEM_A,
    SYSTEM_C,
    UNKNOWN,
    citations,
    readable_ranges,
    request,
    source_block,
)
from daleel.answer.llm import GENERATOR, Reply, Request
from daleel.answer.measure import (
    citation_measures,
    declined,
    declines,
    names_gap,
    numbers,
    states_number,
)
from daleel.retrieve.dense import DOCUMENT, QUESTION, VectorStore
from daleel.retrieve.rerank import ScoreStore

GUIDE = "student_guide_2025"
TATWEEL = chr(0x0640)
# The letter the calendar marks Hijri dates with.
HIJRI = chr(0x0647)


def chunk(chunk_id: str, page: int, text: str, **fields: object) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": GUIDE,
        "doc_title": "دليل الطالب",
        "page": page,
        "section_heading": fields.get("heading", "الرسوم الدراسية"),
        "clause_no": fields.get("clause"),
        "text": text,
    }


SOURCES = [
    chunk("fee", 28, "الطالب السعودي: 250 للوحدة الدراسية", clause="1"),
    chunk("gpa", 40, "ألا يقل المعدل التراكمي عن 3.75 من 4.00"),
    chunk("absence", 30, "الحرمان عند تجاوز الغياب 20%"),
]


def test_arm_c_numbers_its_sources_and_says_where_each_comes_from() -> None:
    asked = request("C", "كم الرسوم؟", SOURCES)
    assert asked.system == SYSTEM_C and asked.model == GENERATOR and asked.purpose == "arm C"
    assert (
        '<source n="1" document="دليل الطالب" page="28" section="الرسوم الدراسية" clause="1">'
        in (asked.prompt)
    )
    assert '<source n="3"' in asked.prompt and "clause" not in source_block(2, SOURCES[1])
    assert asked.prompt.endswith("Question: كم الرسوم؟")


def test_a_source_loses_its_stretching_strokes_and_keeps_its_spelling() -> None:
    stretched = chunk("x", 1, f"مـ{TATWEEL * 3}ن الطالبُ", heading=f"الإنقطـ{TATWEEL}اع")
    block = source_block(1, stretched)
    assert "من الطالبُ" in block and 'section="الإنقطاع"' in block


def test_a_source_says_when_and_whom_its_document_governs() -> None:
    dated = {**SOURCES[0], "effective": "2025", "scope": "commission"}
    block = source_block(1, dated)
    assert (
        'date="2025"' in block and 'scope="جميع كليات ومعاهد الهيئة الملكية للجبيل وينبع"' in block
    )


def test_calendar_ranges_printed_right_to_left_are_written_first_day_first() -> None:
    printed = f"النهائية 2026/12/31-20م، 1448/07/22-11{HIJRI}، وبداية الدراسة 2026/08/23م"
    assert readable_ranges(printed) == (
        f"النهائية 2026/12/20 - 2026/12/31م، 1448/07/11 - 1448/07/22{HIJRI}، "
        "وبداية الدراسة 2026/08/23م"
    )


def test_arm_a_asks_without_sources_and_says_how_to_decline() -> None:
    asked = request("A", "كم الرسوم؟")
    assert asked.system == SYSTEM_A and "<source" not in asked.prompt
    assert UNKNOWN["ar"] in SYSTEM_A and DECLINE["ar"] in SYSTEM_C and GAP["en"] in SYSTEM_C
    with pytest.raises(ValueError, match="no arm 'B'"):
        request("B", "q")


def test_citations_are_read_in_either_script_and_in_lists() -> None:
    cited = citations("الرسوم 250 [1]. المعدل 3.75 [٢][1]، والغياب [3، 7].", sources=3)
    assert cited.numbers == (1, 2, 3) and cited.unknown == (7,)
    assert cited.chunks(SOURCES) == ["fee", "gpa", "absence"]
    assert citations("no citation, and [a] is not one").numbers == ()


def test_declining_and_naming_a_gap_are_found_whatever_the_spelling() -> None:
    assert declined("لا تُجيب الوثائق المتاحة عن هذا السؤال")
    assert declined("I do not know a reliable answer to this question.")
    assert declined("   ")
    assert not declined("الرسوم 250 [1].")
    assert names_gap("الرسوم 250 [1]. لا تذكر الوثائق المتاحة مبلغ المواقف.")
    assert names_gap("The fee is 250 [1]. The available documents do not say when it is paid.")
    assert not names_gap("الرسوم 250 [1].")


def test_saying_only_what_is_missing_with_nothing_cited_is_declining() -> None:
    assert declines("لا تذكر الوثائق المتاحة مكافأة أرامكو.", citations("لا شيء"))
    partial = "الطلب خلال أسبوع [1]. لا تذكر الوثائق المتاحة خطوات الإلغاء."
    assert not declines(partial, citations(partial))
    assert declines(DECLINE["ar"], citations(DECLINE["ar"]))


def test_numbers_are_read_in_either_script_with_their_separators() -> None:
    arabic = str.maketrans("0123456789.", "".join(chr(0x0660 + d) for d in range(10)) + chr(0x066B))
    written = f"المعدل {'3.75'.translate(arabic)} والرسوم 1,100 والغياب {'20'.translate(arabic)}%"
    assert numbers(written) == [3.75, 1100.0, 20.0]
    assert states_number("ألا يقل عن 3.75 من 4.00", 3.75)
    assert numbers("خلال خمسة أيام عمل، وفي الأسبوع السابع") == [5.0, 7.0]
    assert numbers("خمسة عشر يوماً، والمادة الحادية عشرة، وخمس وعشرون") == [15.0, 11.0, 25.0]
    assert states_number("تقدم الطلب خلال عشرة أيام", 10)
    assert not states_number("تقدم الطلب خلال أسبوع [5].", 5)
    assert not states_number("ألا يقل عن 3.5", 3.75)


def test_citations_are_measured_against_the_quotes_they_should_hold() -> None:
    units = [frozenset({"fee"}), frozenset({"gpa", "other"})]
    both = citation_measures(citations("[1][3]"), SOURCES, units)
    assert both == {"precision": 0.5, "recall": 0.5}
    assert citation_measures(citations("none"), SOURCES, units) == {"precision": 0.0, "recall": 0.0}


# --- scripts/run_arms.py ------------------------------------------------------------


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def gold(qid: str, text: str, status: str, **fields: object) -> dict:
    return {
        "qid": qid,
        "type": fields.get("type", "numeric"),
        "lang": "ar",
        "split": "dev",
        "question": text,
        "answerability_status": status,
        "answer_numeric": fields.get("numeric"),
        "source_refs": fields.get("refs", []),
    }


FEE_REF = {"doc_id": GUIDE, "pdf_page": 28, "role": "answer_evidence", "quotes": ["250"]}
QUESTIONS = [
    gold("g1", "كم رسوم الوحدة للطالب السعودي؟", "supported", numeric=250, refs=[FEE_REF]),
    gold("g2", "كم تدفع أرامكو مكافأة؟", "outside_corpus", type="out_of_scope"),
]


def stored(tmp_path: Path, scores: dict[str, list[float]]) -> list[str]:
    """The chunks, the questions, their vectors and reranker scores, on disk, as
    arguments to the run: `scores` gives each question's score for each source."""
    dense = tmp_path / "dense"
    store = VectorStore()
    store.add(DOCUMENT, [c["text"] for c in SOURCES], [unit(1, 0), unit(0, 1), unit(1, 1)])
    store.add(QUESTION, [q["question"] for q in QUESTIONS], [unit(1, 0.1), unit(0, 1)])
    store.save(dense / "bge-m3.npz")
    reranker = ScoreStore()
    for q in QUESTIONS:
        reranker.add(q["question"], [c["text"] for c in SOURCES], scores[q["qid"]])
    reranker.save(tmp_path / "rerank.json")
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", SOURCES))]
    argv += ["--gold", str(write_lines(tmp_path / "gold.jsonl", QUESTIONS))]
    argv += ["--dense", str(dense), "--rerank", str(tmp_path / "rerank.json")]
    return [*argv, "--out", str(tmp_path / "runs"), "--show", "1"]


def saved(tmp_path: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "runs" / name).read_text().splitlines()]


def test_the_run_answers_with_both_arms_and_measures_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    answers = {
        ("arm A", "g1"): "لا أعرف إجابة موثوقة عن هذا السؤال.",
        ("arm C", "g1"): "رسوم الوحدة الدراسية للطالب السعودي 250 [1].",
        ("arm A", "g2"): "تدفع أرامكو 3000 ريال.",
        ("arm C", "g2"): "لا تجيب الوثائق المتاحة عن هذا السؤال.",
    }

    class FakeClient:
        asked = 0

        def ask(self, asked):
            qid = "g1" if "الوحدة" in asked.prompt else "g2"
            return Reply(answers[(asked.purpose, qid)], "STOP", "v1", 100, 20, 5, 0.5)

    run_arms = script("run_arms")
    monkeypatch.setattr(run_arms, "Client", FakeClient)
    argv = stored(tmp_path, {"g1": [2.0, 1.0, 0.0], "g2": [2.0, 1.0, 0.0]})
    assert run_arms.main(argv) == 0
    out = capsys.readouterr().out
    assert "| arm A | 0/1 | - |" in out
    assert "| arm C | 1/1 | 1/1 | 1.00 | 1.00 | 1/1 | 1/1 |" in out
    records = saved(tmp_path, "C-dev.jsonl")
    assert records[0]["sources"][0] == "fee" and records[0]["cited"] == [1]
    assert records[1]["declined"] and records[1]["behavior"] == "refuse"


def test_the_flow_runs_as_an_arm_with_its_retries_and_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    # g1 is answered without a citation, then with one. g2's sources score far
    # below the threshold, so the model is never asked about it.
    answers = ["رسوم الوحدة الدراسية للطالب السعودي 250.", "رسوم الوحدة 250 [1]."]

    sent: list[Request] = []

    class FakeClient:
        asked = 0

        def ask(self, asked: Request) -> Reply:
            sent.append(asked)
            return Reply(answers[len(sent) - 1], "STOP", "v1", 100, 20, 5, 1.0)

    run_arms = script("run_arms")
    monkeypatch.setattr(run_arms, "Client", FakeClient)
    argv = stored(tmp_path, {"g1": [2.0, 1.0, 0.0], "g2": [-6.0, -7.0, -8.0]})
    assert run_arms.main(["--arms", "flow", *argv]) == 0
    out = capsys.readouterr().out
    # answered, cited, precision, recall, numbers, refused, gap named, declined
    # wrongly, gap on answerable, retried, refused unasked, tokens in and out.
    assert (
        "| arm flow | 1/1 | 1/1 | 1.00 | 1.00 | 1/1 | 1/1 | - | 0/1 | 0/1 | 1/2 | 1/2 | 100 | 25 |"
        in out
    )
    assert [asked.purpose for asked in sent] == ["arm C", "arm C"]
    first, second = saved(tmp_path, "flow-dev.jsonl")
    assert first["arm"] == "flow" and first["outcome"] == "answered" and first["attempts"] == 2
    assert first["cited"] == [1] and first["sources"][0] == "fee" and first["numeric_exact"]
    assert first["tokens_in"] == 200 and first["tokens_out"] == 50 and first["seconds"] == 2.0
    assert second["outcome"] == "refused" and second["reason"] == "low confidence"
    assert second["attempts"] == 0 and second["tokens_in"] == 0 and second["finish"] is None
    assert second["answer"] == DECLINE["ar"] and second["declined"]


def test_arms_a_and_c_record_one_attempt_and_no_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Callable[[str], ModuleType],
) -> None:
    class FakeClient:
        asked = 0

        def ask(self, asked: Request) -> Reply:
            return Reply("رسوم الوحدة 250 [1].", "STOP", "v1", 100, 20, 5, 0.5)

    run_arms = script("run_arms")
    monkeypatch.setattr(run_arms, "Client", FakeClient)
    argv = stored(tmp_path, {"g1": [2.0, 1.0, 0.0], "g2": [2.0, 1.0, 0.0]})
    assert run_arms.main(["--arms", "C", *argv]) == 0
    for record in saved(tmp_path, "C-dev.jsonl"):
        assert record["attempts"] == 1 and record["outcome"] is None and record["reason"] is None
