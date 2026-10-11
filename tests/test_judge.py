"""Tests for the judge, for drawing answers to label, and for the judge's runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from daleel.answer.llm import JUDGE, Reply, Request
from daleel.eval.judge import (
    JUDGE_TOKENS,
    LABELS,
    RUBRIC,
    RUBRIC_VERSION,
    case,
    evidence,
    hand_label,
    judge,
    judge_request,
    verdict,
)

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
    assert asked.max_tokens == JUDGE_TOKENS
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


def test_a_judged_answer_carries_the_label_and_the_reasoning() -> None:
    def ask(asked: Request) -> Reply:
        assert asked.purpose == "judge"
        return Reply("Points: the fee. VERDICT: correct", "STOP", "gemma-v", 900, 40, 0, 2.0)

    found = judge(question(), record("الرسوم 250 [1].", [1], ["fee"]), CHUNKS, ask)
    assert found == {
        "judge": "correct",
        "judge_rubric": RUBRIC_VERSION,
        "judge_reasoning": "Points: the fee. VERDICT: correct",
        "judge_finish": "STOP",
        "judge_model": "gemma-v",
    }


def test_a_hand_label_is_read_in_full_or_by_its_first_letter() -> None:
    assert hand_label(" Correct ") == "correct" and hand_label("H") == "hallucinated"
    assert hand_label("p") == "partial" and hand_label("i") == "incorrect"
    assert hand_label("") is None and hand_label("right") is None


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


# --- scripts/judge_agreement.py and scripts/judge_answers.py ------------------------

VERDICTS = {
    "answer g1": "VERDICT: correct",
    "answer g2": "Some reasoning.\nVERDICT: partial",
    "answer g3": "I cannot decide.",
    "invented g1": "VERDICT: hallucinated",
    "invented g2": "VERDICT: incorrect",
    "invented g3": "**VERDICT:** hallucinated",
}


class FakeJudge:
    asked = 0

    def ask(self, asked: Request) -> Reply:
        FakeJudge.asked += 1
        answer = asked.prompt.rsplit("<answer>\n", 1)[1].split("\n</answer>")[0]
        return Reply(VERDICTS[answer], "STOP", "gemma-v", 900, 40, 0, 2.0)


def judged_files(tmp_path: Path) -> list[str]:
    gold = [
        question("g1", type="numeric"),
        question("g2", type="numeric"),
        {**question("g3", "outside_corpus", refs=[]), "type": "out_of_scope"},
    ]
    behaviors = {"g1": "answer", "g2": "answer", "g3": "refuse"}
    for q in gold:
        q.setdefault("type", "numeric")
    runs = tmp_path / "runs"
    runs.mkdir()
    for arm, prefix in (("A", "invented"), ("flow", "answer")):
        rows = [
            record(
                f"{prefix} {q['qid']}",
                [],
                [],
                qid=q["qid"],
                arm=arm,
                type=q["type"],
                behavior=behaviors[q["qid"]],
            )
            for q in gold
        ]
        write_lines(runs / f"{arm}-dev.jsonl", rows)
    argv = ["--chunks", str(write_lines(tmp_path / "chunks.jsonl", list(CHUNKS.values())))]
    return [*argv, "--gold", str(write_lines(tmp_path / "gold.jsonl", gold))]


def write_sheet(folder: Path, labels: list[str], key: list[tuple[str, str]]) -> None:
    folder.mkdir(exist_ok=True)
    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item", "label", "note"])
        writer.writerows(
            [n, label, "hard one" if n == 2 else ""] for n, label in enumerate(labels, 1)
        )
    write_lines(
        folder / "key.jsonl",
        [
            {"item": n, "run": f"{arm}-dev.jsonl", "arm": arm, "qid": qid}
            for n, (arm, qid) in enumerate(key, 1)
        ],
    )


def test_t9_puts_the_hand_labels_against_the_judge_and_lists_where_they_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    argv = judged_files(tmp_path)
    key = [("flow", "g1"), ("flow", "g2"), ("A", "g1"), ("flow", "g3"), ("A", "g3")]
    write_sheet(tmp_path / "labels", ["C", "correct", "h", "incorrect", "hallucinated"], key)
    judge_agreement = script("judge_agreement")
    monkeypatch.setattr(judge_agreement, "Client", FakeJudge)
    argv += ["--labels", str(tmp_path / "labels"), "--runs", str(tmp_path / "runs")]
    assert judge_agreement.main(argv) == 0
    out = capsys.readouterr().out
    # Item 4 got no verdict, so four answers are compared.
    assert "| 4 answers | judge: correct | judge: partial |" in out
    assert "| human: correct | 1 | 1 | 0 | 0 |" in out
    assert "| human: hallucinated | 0 | 0 | 0 | 2 |" in out
    assert "the labels agree on 3 of 4" in out and "1 answers had no verdict" in out
    assert f"Judge {JUDGE}, rubric {RUBRIC_VERSION}." in out
    assert "arm A: agree on 2 of 2" in out and "arm flow: agree on 1 of 2" in out
    assert "- item 2 (g2, arm flow): human correct, judge partial. Note: hard one" in out
    assert "- item 4 (g3, arm flow): human incorrect, judge None" in out
    saved = [json.loads(line) for line in (tmp_path / "labels" / "judged.jsonl").open()]
    assert [s["judge"] for s in saved] == [
        "correct",
        "partial",
        "hallucinated",
        None,
        "hallucinated",
    ]


def test_t9_waits_until_every_answer_is_labelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    argv = judged_files(tmp_path)
    write_sheet(tmp_path / "labels", ["correct", "", "maybe"], [("flow", "g1")] * 3)
    judge_agreement = script("judge_agreement")
    FakeJudge.asked = 0
    monkeypatch.setattr(judge_agreement, "Client", FakeJudge)
    argv += ["--labels", str(tmp_path / "labels"), "--runs", str(tmp_path / "runs")]
    assert judge_agreement.main(argv) == 4
    err = capsys.readouterr().err
    assert "item 2: ''" in err and "item 3: 'maybe'" in err and FakeJudge.asked == 0


def test_a_judged_run_counts_each_label_overall_and_by_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    argv = judged_files(tmp_path)
    judge_answers = script("judge_answers")
    monkeypatch.setattr(judge_answers, "Client", FakeJudge)
    runs = [str(tmp_path / "runs" / "A-dev.jsonl"), str(tmp_path / "runs" / "flow-dev.jsonl")]
    assert judge_answers.main([*argv, "--runs", *runs]) == 0
    out = capsys.readouterr().out
    # answers, correct, partial, incorrect, hallucinated, no verdict, accuracy, invented
    # outside the corpus.
    assert "| A-dev | 3 | 0 | 0 | 1 | 2 | 0 | 0.00 | 1/1 |" in out
    assert "| flow-dev | 3 | 1 | 1 | 0 | 0 | 1 | 0.33 | 0/1 |" in out
    assert "| numeric | 2 | 1 | 1 | 0 | 0 | 0 |" in out
    saved = [json.loads(line) for line in (tmp_path / "runs" / "flow-dev-judged.jsonl").open()]
    assert [s["judge"] for s in saved] == ["correct", "partial", None]
