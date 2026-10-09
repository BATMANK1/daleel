"""Measures of an answer that need no judge.

Whether it declined, whether it named a gap, whether its citations are well
formed and hold the gold evidence, and whether it states a numeric question's
number exactly. Accuracy itself is judged elsewhere, against the gold answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from daleel.answer.arms import DECLINE, GAP, UNKNOWN, Citations
from daleel.normalize.arabic import for_comparison

# The Arabic decimal separator.
DECIMAL = chr(0x066B)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _has(text: str, phrases: Mapping[str, str]) -> bool:
    folded = for_comparison(text)
    return any(for_comparison(phrase).rstrip(".") in folded for phrase in phrases.values())


def declined(answer: str) -> bool:
    """Whether the answer declines, in either arm's fixed words, or is empty."""
    return not answer.strip() or _has(answer, DECLINE) or _has(answer, UNKNOWN)


def names_gap(answer: str) -> bool:
    """Whether the answer says, in the fixed words, what its sources leave out."""
    return _has(answer, GAP)


def numbers(text: str) -> list[float]:
    """The numbers written in a text, in either script, with thousands separators
    and the Arabic decimal separator."""
    folded = for_comparison(text).replace(DECIMAL, ".")
    return [float(found) for found in _NUMBER.findall(_THOUSANDS.sub("", folded))]


def states_number(answer: str, expected: float) -> bool:
    """Whether the answer writes the expected number anywhere."""
    return any(abs(found - expected) < 1e-9 for found in numbers(answer))


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
