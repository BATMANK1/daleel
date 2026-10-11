"""The judge: one label for an answer, against the gold set's reference.

An answer is labelled correct, partial, incorrect or hallucinated, by the
definitions in LABELS. The judge, a language model, reads what a person
labelling the answer by hand reads: the question, what a correct answer does,
the reference answer, the quotes that hold it, the sources the answer cites,
and the answer. It reasons first and ends with one fixed line, VERDICT: and
the label, which `verdict` reads.

The judge is a different model from the one that answers (llm.JUDGE), and it
is validated against labels made by hand before any arm's result is trusted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from daleel.answer.arms import source_block
from daleel.answer.llm import JUDGE, Reply, Request
from daleel.eval.gold import fragments

# The judge's longest reply. Gemma may think before it answers, and at 2,048
# tokens its thinking ran out the reply on 3 of 40 answers, leaving no verdict.
JUDGE_TOKENS = 8192
# Changed when the rubric is: 1 was checked against hand labels first, and 2
# settles how declining and refusing count, and catches steps out of order.
RUBRIC_VERSION = 2

# Where answers drawn for labelling by hand, and their labels, are kept.
HAND_LABELS = Path("data/interim/labels")

# Each label and when it applies. The person labelling by hand reads these
# same words.
LABELS = {
    "correct": ("It makes every point that counts, and nothing in it is false or invented."),
    "partial": (
        "It makes some of the points that count but not all, and invents nothing. "
        "Also partial: an answer that makes every point but says the documents lack "
        "something the evidence shows they contain, or that decides between sources "
        "when a correct answer gives each."
    ),
    "incorrect": (
        "It makes none of the points that count, and invents nothing: for example, "
        "it declines a question the documents answer, or answers a different question."
    ),
    "hallucinated": (
        "It states as fact a specific detail, such as a number, date, period, "
        "percentage, condition, step or the order of steps, office or name, that "
        "contradicts the reference or the evidence, or that neither they nor any "
        "source it cites supports. A detail stated before the answer declines still "
        "counts. Answering a question the documents do not answer with such details "
        "is hallucinated. This label wins over the others."
    ),
}

POINTS = (
    "Read the reference answer as a list of points: each fact it gives, and each "
    "thing it says the documents do not contain. The points that count are those "
    "that answer what the question asks; what a correct answer does says which "
    "matter most. For a question the documents do not answer at all, the one point "
    "that counts is that they do not contain the answer; the reference's other "
    "remarks do not count. An answer that declines, or says the documents do not "
    "answer, makes only the points that say the documents do not contain something. "
    "Details beyond the reference are fine where the evidence or a "
    "source the answer cites supports them. Wording, length, language and the "
    "form of citations do not count here; they are measured elsewhere."
)

RUBRIC = "\n".join(
    [
        "You grade one answer to a student's question about the regulations of the "
        "colleges and institutes of the Royal Commission for Jubail and Yanbu, "
        "against a reference answer written from the documents.",
        "",
        POINTS,
        "",
        "Give exactly one of these labels:",
        *(f"- {label}: {definition}" for label, definition in LABELS.items()),
        "",
        "First list the points that count and check each against the answer. Then "
        "check every specific detail the answer states. Be brief. End with exactly "
        "one line, VERDICT: followed by the label, such as VERDICT: partial.",
    ]
)

_VERDICT = re.compile(r"VERDICT\W*(" + "|".join(LABELS) + r")\b", re.IGNORECASE)


def evidence(question: Mapping[str, Any]) -> list[str]:
    """The quotes of the pages that hold the question's answer."""
    return [
        quote
        for ref in question.get("source_refs", [])
        if ref.get("role") == "answer_evidence"
        for quote in fragments(ref.get("quotes", []))
    ]


def cited_sources(record: Mapping[str, Any], chunks: Mapping[str, Mapping[str, Any]]) -> str:
    """The sources an answer cites, numbered as it cites them, or "none"."""
    blocks = [
        source_block(number, chunks[record["sources"][number - 1]])
        for number in record.get("cited", [])
    ]
    return "\n".join(blocks) or "none"


def case(
    question: Mapping[str, Any],
    record: Mapping[str, Any],
    chunks: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """What the judge and a person labelling by hand both read about one answer,
    by the name of each part."""
    quotes = evidence(question)
    return {
        "question": question["question"],
        "expected_behavior": question.get("expected_behavior") or "",
        "reference_answer": question["answer"],
        "evidence": "\n".join(f"- {quote}" for quote in quotes) or "none",
        "cited_sources": cited_sources(record, chunks),
        "answer": record["answer"],
    }


def judge_request(
    question: Mapping[str, Any],
    record: Mapping[str, Any],
    chunks: Mapping[str, Mapping[str, Any]],
    model: str = JUDGE,
) -> Request:
    """What the judge is asked about one answer.

    The rubric goes in the prompt rather than a system instruction, since Gemma
    models on this API have not always accepted one. Temperature 0, so that
    the judge's labels vary as little as it allows.
    """
    parts = case(question, record, chunks).items()
    prompt = "\n\n".join([RUBRIC, *(f"<{name}>\n{text}\n</{name}>" for name, text in parts)])
    return Request(model, prompt, max_tokens=JUDGE_TOKENS, temperature=0.0, purpose="judge")


def verdict(text: str) -> str | None:
    """The label on the judge's last VERDICT line, or None if it gave none."""
    found = _VERDICT.findall(text)
    return found[-1].lower() if found else None


def judge(
    question: Mapping[str, Any],
    record: Mapping[str, Any],
    chunks: Mapping[str, Mapping[str, Any]],
    ask: Callable[[Request], Reply],
    model: str = JUDGE,
) -> dict[str, Any]:
    """The judge's label for one answer, with its reasoning, as fields to add to
    the answer's record. The label is None where the judge gave no verdict."""
    reply = ask(judge_request(question, record, chunks, model))
    return {
        "judge": verdict(reply.text),
        "judge_rubric": RUBRIC_VERSION,
        "judge_reasoning": reply.text,
        "judge_finish": reply.finish,
        "judge_model": reply.model_version,
    }


def hand_label(text: str) -> str | None:
    """A label as a person writes it, in full or by its first letter, in any case;
    None if it is none of the labels."""
    word = text.strip().lower()
    return next((label for label in LABELS if word in (label, label[0])), None) if word else None
