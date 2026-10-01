"""Put what an OCR engine read in Arabic reading order.

An engine that finds text boxes and reads each one separately, as PaddleOCR
does, leaves the order of the page to its caller. PaddleOCR sorts boxes top to
bottom and then left to right, the order of Latin script: on an Arabic line
split into several boxes, that puts the end of the line first. This module
reads such a page the way the annotation guidelines do (rule 4.1): rows top to
bottom, and each row right to left.

Rows are found from geometry alone. Two boxes share a row when each one's
vertical middle lies within the other's height, so neighbouring lines stay
apart while boxes of one line that sit a little higher or lower join. Columns
are not detected: two blocks side by side are read row by row across both.

A layout model, as dots.mocr is, orders the blocks it finds itself, and mostly
well, but it reads blocks side by side left to right, as in the pages it
learned from. The library deck's lending policy (page 8) has two columns, and
dots.mocr read the left one first: the policy's second part before the slide's
title and its first part, on the right. The academic calendar's rows of cards
(T2) came back each read left to right, against the order of their dates. So
the model's order is split into the parts it read one after another, and two
parts it read left to right, side by side, are read right to left. Nothing
else moves: on the other 15 slides and on 8 pages of five other documents read
for T2, the order is the model's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from daleel.ocr.engine import Block
from daleel.ocr.engine import Box as BlockBox


@dataclass(frozen=True)
class Box:
    """A piece of text and the upright rectangle it was read from, in pixels."""

    text: str
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def around(cls, text: str, points: Iterable[Sequence[float]]) -> Box:
        """The upright box around a polygon given as (x, y) points."""
        xs, ys = zip(*((float(point[0]), float(point[1])) for point in points), strict=True)
        return cls(text, min(xs), min(ys), max(xs), max(ys))

    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2

    def spans(self, y: float) -> bool:
        return self.top <= y <= self.bottom


def same_row(a: Box, b: Box) -> bool:
    """Whether each box's vertical middle lies within the other's height."""
    return a.spans(b.middle) and b.spans(a.middle)


def rows(boxes: Iterable[Box]) -> list[list[Box]]:
    """Boxes grouped into rows, top to bottom, each row ordered right to left.

    A box joins the row above it when it shares a row with that row's tallest
    box; otherwise it starts a new row. The tallest box is the one most likely
    to be a line of text: a small box read from a background graphic, sitting
    a little above a line, would split the line if it anchored the row.
    """
    found: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: b.middle):
        if found and same_row(max(found[-1], key=lambda b: b.bottom - b.top), box):
            found[-1].append(box)
        else:
            found.append([box])
    return [sorted(row, key=lambda b: b.right, reverse=True) for row in found]


def arabic_reading_order(boxes: Iterable[Box]) -> str:
    """The boxes' text as a page: one line per row, right to left within it."""
    lines = (" ".join(box.text for box in row if box.text) for row in rows(boxes))
    return "\n".join(line for line in lines if line)


# Blocks a layout model ordered

# What a page repeats at its edges keeps the model's place: a page number in a
# corner sits beside the content without being part of reading it.
EDGES = frozenset({"Page-header", "Page-footer"})

_ARABIC_LETTER = re.compile("[\u0621-\u063a\u0641-\u064a]")
_LATIN_LETTER = re.compile("[A-Za-z]")


def mostly_arabic(blocks: Iterable[Block]) -> bool:
    """Whether the blocks' text has more Arabic letters than Latin ones."""
    text = "".join(block.text for block in blocks)
    return len(_ARABIC_LETTER.findall(text)) > len(_LATIN_LETTER.findall(text))


def _extents(boxes: Sequence[BlockBox]) -> list[BlockBox]:
    """The left, top, right and bottom of boxes[:n] taken together, for n from 1."""
    found: list[BlockBox] = []
    for left, top, right, bottom in boxes:
        if found:
            last = found[-1]
            left, top = min(left, last[0]), min(top, last[1])
            right, bottom = max(right, last[2]), max(bottom, last[3])
        found.append((left, top, right, bottom))
    return found


def _in_order(places: Sequence[int], blocks: Sequence[Block]) -> list[int]:
    """The places of blocks in the model's order, with every two parts read left to right swapped.

    The model's order is split where it moves on from one part of the page to
    another: between columns, where everything read so far lies wholly left
    or wholly right of everything after and beside it, or between rows, where
    everything read so far lies wholly above everything after. Of the places
    to split, the one with the widest gap is taken, which keeps a heading over
    one of two columns with its column and a row of cards in its row. Two parts
    read left first are swapped when both hold text, since a picture beside
    text changes nothing anyone reads. Each part is split again in turn, and a
    part that cannot be split keeps the model's order.
    """
    if len(places) < 2:
        return list(places)
    boxes = [blocks[place].box for place in places]
    before = _extents(boxes)
    after = _extents(boxes[::-1])[::-1]
    # The widest gap, whether it lies between rows, where it falls, and
    # whether the part before it lies to the left.
    best: tuple[float, bool, int, bool] | None = None
    for middle in range(1, len(places)):
        left, top, right, bottom = before[middle - 1]
        next_left, next_top, next_right, next_bottom = after[middle]
        cuts = [(next_top - bottom, True, False)]
        if top < next_bottom and next_top < bottom:
            cuts += [(next_left - right, False, True), (left - next_right, False, False)]
        for gap, between_rows, left_first in cuts:
            if gap >= 0 and (best is None or (gap, between_rows) > best[:2]):
                best = (gap, between_rows, middle, left_first)
    if best is None:
        return list(places)
    _, _, middle, left_first = best
    first, second = places[:middle], places[middle:]
    if left_first and all(any(blocks[place].text for place in part) for part in (first, second)):
        first, second = second, first
    return _in_order(first, blocks) + _in_order(second, blocks)


def block_order(blocks: Sequence[Block]) -> list[int]:
    """The order to read a layout model's blocks in, as their places in the model's order.

    Two parts of the page the model read left to right, side by side, are read
    right to left. Page headers and footers keep the model's places. A page
    mostly in Latin script, or with a block the model gave no box, keeps the
    model's order.
    """
    order = list(range(len(blocks)))
    body = [place for place, block in enumerate(blocks) if block.category not in EDGES]
    if not mostly_arabic(blocks) or any(blocks[place].box is None for place in body):
        return order
    for slot, place in zip(body, _in_order(body, blocks), strict=True):
        order[slot] = place
    return order
