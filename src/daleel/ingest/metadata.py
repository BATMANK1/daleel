from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class PdfMetadata:
    producer: str
    creator: str


def metadata_value(metadata: dict | None, key: str) -> str:
    """Return one metadata field as a clean string, or "" if it is missing.

    PDF Info dictionaries are unreliable: keys may be absent and values may
    arrive as bytes in any encoding.
    """
    if not metadata:
        return ""
    raw = metadata.get(key)
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def metadata_from(pdf) -> PdfMetadata:
    """Read producer and creator from a PDF that is already open."""
    producer = metadata_value(pdf.metadata, "Producer")
    creator = metadata_value(pdf.metadata, "Creator")

    return PdfMetadata(producer=producer, creator=creator)


def read_metadata(path: Path) -> PdfMetadata:
    """Open a PDF just to read its producer and creator."""
    with pdfplumber.open(path) as pdf:
        return metadata_from(pdf)
