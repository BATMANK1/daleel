"""Measures of an answer that need no judge.

Whether it declined, whether it named a gap, whether its citations are well
formed and hold the gold evidence, and whether it states a numeric question's
number exactly. Accuracy itself is judged elsewhere, against the gold answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from daleel.answer.arms import CITATION, DECLINE, GAP, UNKNOWN, Citations
from daleel.normalize.arabic import for_comparison

# The Arabic decimal separator.
DECIMAL = chr(0x066B)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Numbers written as words, cardinal or ordinal, either gender, with or without
# the article. A run of them adds up: خمسة عشر is 15, الحادية عشرة 11, خمس
# وعشرون 25. Duals, such as قاعتان for two rooms, are not read.
_WORDS = {
    for_comparison(spelling): value
    for spellings, value in (
        ("واحد واحدة الأول الأولى الحادي الحادية", 1),
        ("اثنان اثنين اثنتان اثنتين الثاني الثانية", 2),
        ("ثلاث ثلاثة الثالث الثالثة", 3),
        ("أربع أربعة الرابع الرابعة", 4),
        ("خمس خمسة الخامس الخامسة", 5),
        ("ست ستة السادس السادسة", 6),
        ("سبع سبعة السابع السابعة", 7),
        ("ثمان ثماني ثمانية الثامن الثامنة", 8),
        ("تسع تسعة التاسع التاسعة", 9),
        ("عشر عشرة العاشر العاشرة", 10),
        ("عشرون عشرين العشرون العشرين", 20),
        ("ثلاثون ثلاثين الثلاثون الثلاثين", 30),
        ("أربعون أربعين الأربعون الأربعين", 40),
        ("خمسون خمسين الخمسون الخمسين", 50),
        ("ستون ستين الستون الستين", 60),
        ("سبعون سبعين السبعون السبعين", 70),
        ("ثمانون ثمانين الثمانون الثمانين", 80),
        ("تسعون تسعين التسعون التسعين", 90),
        ("مائة مئة المائة المئة", 100),
    )
    for spelling in spellings.split()
}


def _has(text: str, phrases: Mapping[str, str]) -> bool:
    folded = for_comparison(text)
    return any(for_comparison(phrase).rstrip(".") in folded for phrase in phrases.values())


def declined(answer: str) -> bool:
    """Whether the answer declines, in either arm's fixed words, or is empty."""
    return not answer.strip() or _has(answer, DECLINE) or _has(answer, UNKNOWN)


def names_gap(answer: str) -> bool:
    """Whether the answer says, in the fixed words, what its sources leave out."""
    return _has(answer, GAP)


def declines(answer: str, cited: Citations) -> bool:
    """Whether the answer declines in substance: in the fixed words, or by saying
    only what the documents leave out, with nothing cited."""
    return declined(answer) or (names_gap(answer) and not cited.numbers)


def _word_value(token: str) -> int:
    """A number word's value, read through a leading و or ب, or 0."""
    if token in _WORDS:
        return _WORDS[token]
    if token[:1] in ("و", "ب") and token[1:] in _WORDS:
        return _WORDS[token[1:]]
    return 0


def numbers(text: str) -> list[float]:
    """The numbers written in a text: in digits of either script, with thousands
    separators and the Arabic decimal separator, and in words."""
    folded = for_comparison(text).replace(DECIMAL, ".")
    found = [float(number) for number in _NUMBER.findall(_THOUSANDS.sub("", folded))]
    run = 0
    # Punctuation ends a run, so that a list of numbers is not added up.
    for token in [*re.findall(r"\w+|[^\w\s]", folded), ""]:
        value = _word_value(token) if not token.isdigit() else 0
        if value:
            run += value
        elif run:
            found.append(float(run))
            run = 0
    return found


def states_number(answer: str, expected: float) -> bool:
    """Whether the answer writes the expected number anywhere outside its citations."""
    return any(abs(found - expected) < 1e-9 for found in numbers(CITATION.sub(" ", answer)))


def citation_measures(
    cited: Citations, sources: Sequence[Mapping[str, Any]], units: Sequence[frozenset[str]]
) -> dict[str, float]:
    """How well an answer's citations match the gold evidence.

    precision: the share of the chunks it cites that hold some gold quote.
    recall: the share of the gold quotes held by some chunk it cites.
    """
    chunks = set(cited.chunks(sources))
    holding = set().union(*units) if units else set()
    precision = len(chunks & holding) / len(chunks) if chunks else 0.0
    recall = sum(bool(unit & chunks) for unit in units) / len(units) if units else 0.0
    return {"precision": precision, "recall": recall}
