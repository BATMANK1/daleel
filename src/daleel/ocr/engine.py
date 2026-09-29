"""What every OCR engine provides, so the evaluation can run any of them alike."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Result:
    """What an engine read from one page image, how long it took, and what it warned about."""

    text: str
    seconds: float
    warnings: str = ""


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
