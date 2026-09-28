"""Score OCR output against hand-made ground truth.

The character error rate (CER) is the fewest single-character insertions,
deletions and substitutions that turn an engine's output into the ground
truth, divided by the ground truth's length. The word error rate (WER) is the
same count over words. Both are reported in two forms:

- raw: the visible text as printed. Only canonically equivalent encodings are
  unified (NFC), invisible characters such as direction marks are dropped, and
  runs of whitespace collapse to one space, since line breaks are layout rather
  than text.
- normalized: both sides also pass through for_comparison, the normalization
  retrieval uses. The gap between the two forms shows how much of an engine's
  error normalization absorbs.

A region's ground truth covers part of a page, while an engine reads the whole
page. A region is scored against the stretch of output that matches it with
the fewest edits: text read outside the region costs nothing, and everything
inside the region must still be accounted for.

A table has no single reading order, so an edit count would mostly measure the
order an engine happened to read its cells in. Tables are scored instead by
the words the engine missed, counted as a bag.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

from daleel.normalize.arabic import for_comparison, strip_invisible


def raw_form(text: str) -> str:
    """The visible text as printed, with canonical encodings unified and whitespace collapsed.

    Invisible characters are display instructions, not text. Tesseract wraps
    Arabic runs on mixed lines in direction marks, and counting those would
    charge it for characters no reader sees.
    """
    return " ".join(strip_invisible(unicodedata.normalize("NFC", text)).split())


def normalized_form(text: str) -> str:
    """The text as retrieval compares it.

    NFC comes first so that a letter an engine writes decomposed, such as yaa
    followed by a combining hamza, is not mistaken for yaa with a diacritic.
    """
    return for_comparison(unicodedata.normalize("NFC", text))


FORMS: dict[str, Callable[[str], str]] = {"raw": raw_form, "normalized": normalized_form}


def words(text: str) -> list[str]:
    """Runs of letters, digits and the marks on them.

    Spaces and punctuation both separate words, so الكلية/المعهد is two words
    whether or not a space follows the slash, and (%70) is the single word 70.
    """
    found: list[str] = []
    current: list[str] = []
    for ch in text:
        if unicodedata.category(ch)[0] in "LMN":
            current.append(ch)
        elif current:
            found.append("".join(current))
            current = []
    if current:
        found.append("".join(current))
    return found


@dataclass(frozen=True)
class Errors:
    """How many edits the output needs, against how long the ground truth is."""

    edits: int
    length: int

    @property
    def rate(self) -> float:
        if self.length == 0:
            raise ValueError("an error rate needs ground truth that is not empty")
        return self.edits / self.length

    def __add__(self, other: Errors) -> Errors:
        # Summing before dividing weights each page by its length, so a short
        # page cannot swing the rate for a whole document.
        return Errors(self.edits + other.edits, self.length + other.length)


def edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """The fewest insertions, deletions and substitutions that turn a into b."""
    # A shared prefix or suffix never needs an edit, and trimming it first makes
    # nearly identical pages cheap to compare.
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a, b = a[start:end_a], b[start:end_b]
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Span:
    """Where a pattern sits inside a text, and how many edits the match needs."""

    start: int
    end: int
    edits: int


def locate(pattern: Sequence[Hashable], text: Sequence[Hashable]) -> Span:
    """Find the stretch of text that pattern matches with the fewest edits.

    This is semi-global alignment: the pattern must be matched from its first
    item to its last, but the match may start and end anywhere in the text.
    When several stretches need equally few edits, the one ending first wins.
    """
    # distance[j]: fewest edits matching the pattern so far against a stretch of
    # text ending at j. origin[j]: where that stretch starts. Before any of the
    # pattern is matched, a stretch may start anywhere at no cost.
    distance = [0] * (len(text) + 1)
    origin = list(range(len(text) + 1))
    for i, p in enumerate(pattern, start=1):
        row, row_origin = [i], [0]
        for j, t in enumerate(text, start=1):
            best, best_origin = distance[j - 1] + (p != t), origin[j - 1]
            if distance[j] + 1 < best:  # the pattern has an item the text lacks
                best, best_origin = distance[j] + 1, origin[j]
            if row[j - 1] + 1 < best:  # the text has an item the pattern lacks
                best, best_origin = row[j - 1] + 1, row_origin[j - 1]
            row.append(best)
            row_origin.append(best_origin)
        distance, origin = row, row_origin
    end = min(range(len(distance)), key=distance.__getitem__)
    return Span(start=origin[end], end=end, edits=distance[end])


def _count(reference: Sequence[Hashable], output: Sequence[Hashable], *, region: bool) -> Errors:
    edits = locate(reference, output).edits if region else edit_distance(reference, output)
    return Errors(edits=edits, length=len(reference))


def char_errors(
    reference: str, output: str, *, form: Callable[[str], str] = raw_form, region: bool = False
) -> Errors:
    """Character errors in an engine's output, with the reference as ground truth.

    With region=True the reference covers only part of the page the output
    was read from, and is matched against the best stretch of it.
    """
    return _count(form(reference), form(output), region=region)


def word_errors(
    reference: str,
    output: str,
    *,
    form: Callable[[str], str] = normalized_form,
    region: bool = False,
) -> Errors:
    """Word errors in an engine's output, with the reference as ground truth."""
    return _count(words(form(reference)), words(form(output)), region=region)


def missed_words(
    reference: str, output: str, *, form: Callable[[str], str] = normalized_form
) -> Errors:
    """Words of the reference found nowhere in the output, ignoring order.

    For tables, whose cells have no single reading order. Words count as a bag:
    a word the table holds twice must appear twice in the output.
    """
    expected = Counter(words(form(reference)))
    found = expected & Counter(words(form(output)))
    total = expected.total()
    return Errors(edits=total - found.total(), length=total)
