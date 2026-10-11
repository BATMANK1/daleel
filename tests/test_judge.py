"""Tests for the judge's request and verdict, and for drawing answers to label."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from daleel.answer.llm import JUDGE
from daleel.eval.judge import LABELS, RUBRIC, case, evidence, judge_request, verdict

GUIDE = "student_guide_2025"
CHUNKS = {
    "fee": {
        "chunk_id": "fee",
        "doc_id": GUIDE,
        "doc_title": "دليل الطالب",
        "page": 28,
        "section_heading": "الرسوم",
        "text": "الطالب السعودي: 250 للوحدة الدراسية",
    },
    "gpa": {
        "chunk_id": "gpa",
        "doc_id": GUIDE,
        "doc_title": "دليل الطالب",
        "page": 40,
        "section_heading": "المعدل",
        "text": "ألا يقل المعدل عن 3.75",
    },
}


def question(qid: str = "g1", status: str = "supported", **fields: object) -> dict:
    return {
        "qid": qid,
        "question": "كم رسوم الوحدة؟",
        "answer": "250 ريالاً للوحدة الدراسية.",
        "expected_behavior": "ذكر المبلغ مع مصدره.",
        "answerability_status": status,
        "behavior": fields.get("behavior", "answer"),
        "source_refs": fields.get(
            "refs",
            [
                {"role": "answer_evidence", "quotes": ["250 للوحدة", ["السعودي", "250"]]},
                {"role": "related_only"},
            ],
        ),
    }


def record(answer: str, cited: list[int], sources: list[str], **fields: object) -> dict:
    return {"answer": answer, "cited": cited, "sources": sources, **fields}


def test_the_judge_reads_the_reference_the_evidence_and_what_the_answer_cites() -> None:
    answer = "الرسوم 250 [2]."
    asked = judge_request(question(), record(answer, [2], ["gpa", "fee"]), CHUNKS)
    assert asked.model == JUDGE and asked.temperature == 0.0 and asked.purpose == "judge"
    assert asked.system == "" and asked.prompt.startswith(RUBRIC)
    for label in LABELS:
        assert f"- {label}: " in asked.prompt
    assert "<reference_answer>\n250 ريالاً للوحدة الدراسية.\n</reference_answer>" in asked.prompt
    assert "<evidence>\n- 250 للوحدة\n- السعودي\n- 250\n</evidence>" in asked.prompt
    # Only the cited source, under the number the answer cites it by.
    assert '<source n="2" document="دليل الطالب" page="28"' in asked.prompt
    assert "3.75" not in asked.prompt
    assert asked.prompt.endswith(f"<answer>\n{answer}\n</answer>")


def test_an_answer_without_sources_or_evidence_says_none() -> None:
    outside = question("g2", "outside_corpus", refs=[{"role": "related_only"}])
    found = case(outside, record("لا أعرف.", [], []), CHUNKS)
    assert found["evidence"] == "none" and found["cited_sources"] == "none"
    assert evidence(outside) == []


def test_the_verdict_is_the_last_one_given_in_any_case_or_markup() -> None:
    assert verdict("The points... VERDICT: partial\n\nVERDICT: Correct") == "correct"
    assert verdict("**VERDICT:** hallucinated") == "hallucinated"
    assert verdict("verdict: INCORRECT.") == "incorrect"
    assert verdict("I think it is correct.") is None
    assert verdict("VERDICT: mostly right") is None


# --- scripts/draw_labels.py ---------------------------------------------------------


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def test_the_draw_takes_each_arm_equally_hides_the_arm_and_can_be_made_again(
    tmp_path: Path, script: Callable[[str], ModuleType]
) -> None:
    behaviors = ["answer"] * 8 + ["refuse", "answer_and_name_the_gap"]
    gold = [{**question(f"g{n}"), "behavior": b} for n, b in enumerate(behaviors)]
    runs = []
    # Arm A cites nothing; the flow cites its one source.
    for arm, cited, sources in (("A", [], []), ("flow", [1], ["fee"])):
        rows = [
            record(
                f"answer {q['qid']}", cited, sources, qid=q["qid"], arm=arm, behavior=q["behavior"]
            )
            for q in gold
        ]
        runs.append(write_lines(tmp_path / f"{arm}-dev.jsonl", rows))
    chunks = write_lines(tmp_path / "chunks.jsonl", list(CHUNKS.values()))
    gold_file = write_lines(tmp_path / "gold.jsonl", gold)
    draw_labels = script("draw_labels")
    argv = ["--runs", *map(str, runs), "--count", "8", "--chunks", str(chunks)]
    argv += ["--gold", str(gold_file)]
    assert draw_labels.main([*argv, "--out", str(tmp_path / "one")]) == 0
    assert draw_labels.main([*argv, "--out", str(tmp_path / "two")]) == 0

    key = [json.loads(line) for line in (tmp_path / "one" / "key.jsonl").open()]
    assert [k["item"] for k in key] == list(range(1, 9))
    assert sorted(k["arm"] for k in key) == ["A"] * 4 + ["flow"] * 4
    # Half of each arm's share goes to questions that are not answered plainly.
    by_qid = {q["qid"]: q["behavior"] for q in gold}
    for arm in ("A", "flow"):
        assert sum(by_qid[k["qid"]] != "answer" for k in key if k["arm"] == arm) == 2
    page = (tmp_path / "one" / "sheet.html").read_text(encoding="utf-8")
    assert page.count("<section>") == 8 and "flow" not in page and '"arm"' not in page
    for label in LABELS:
        assert f"<dt>{label}</dt>" in page
    with (tmp_path / "one" / "labels.csv").open(encoding="utf-8") as file:
        assert list(csv.reader(file))[:2] == [["item", "label", "note"], ["1", "", ""]]
    for name in ("sheet.html", "labels.csv", "key.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
