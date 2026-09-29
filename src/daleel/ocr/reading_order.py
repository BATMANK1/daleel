"""Put detected text boxes in Arabic reading order.

An engine that finds text boxes and reads each one separately, as PaddleOCR
does, leaves the order of the page to its caller. PaddleOCR sorts boxes top to
bottom and then left to right, the order of Latin script: on an Arabic line
split into several boxes, that puts the end of the line first. This module
reads a page the way the annotation guidelines do (rule 4.1): rows top to
bottom, and each row right to left.

Rows are found from geometry alone. Two boxes share a row when each one's
vertical middle lies within the other's height, so neighbouring lines stay
apart while boxes of one line that sit a little higher or lower join. Columns
are not detected: two blocks side by side are read row by row across both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


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

    A box joins the row above it when it shares a row with that row's first
    box; otherwise it starts a new row.
    """
    found: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: b.middle):
        if found and same_row(found[-1][0], box):
            found[-1].append(box)
        else:
            found.append([box])
    return [sorted(row, key=lambda b: b.right, reverse=True) for row in found]


def arabic_reading_order(boxes: Iterable[Box]) -> str:
    """The boxes' text as a page: one line per row, right to left within it."""
    lines = (" ".join(box.text for box in row if box.text) for row in rows(boxes))
    return "\n".join(line for line in lines if line)
