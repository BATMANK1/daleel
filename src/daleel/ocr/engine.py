"""What every OCR engine provides, so the evaluation can run any of them alike."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Block:
    """One block of a page's layout, from an engine that reads layout."""

    category: str
    # The text the block prints: none for a picture, a line per row for a table.
    text: str
    # Left, top, right and bottom as fractions of the image's width and height,
    # from its top left corner, so the box fits the page at any resolution.
    # None when the engine gave no box.
    box: Box | None = None
    # A table as the engine wrote it, in HTML, so its cells stay apart.
    html: str = ""


@dataclass(frozen=True)
class Result:
    """What an engine read from one page image, how long it took, and what it warned about."""

    text: str
    seconds: float
    warnings: str = ""
    # The engine's whole answer, when it says more than the text: a served
    # model's JSON response, with the page's layout and the tokens it took.
    response: str = ""
    # The page's blocks in reading order, from an engine that reads layout.
    blocks: tuple[Block, ...] = ()
    # The answer stopped at a length limit, so a larger limit could add to it.
    truncated: bool = False


class Engine(Protocol):
    """One engine, configured once: everything that can change its output is fixed when built."""

    def tag(self) -> str:
        """A short name for this configuration, used for the folder its output goes to."""
        ...

    def describe(self) -> str:
        """The versions, models and settings behind every result, for a report's header."""
        ...

    def recognize(self, png: bytes) -> Result:
        """Read the text of one page image."""
        ...
