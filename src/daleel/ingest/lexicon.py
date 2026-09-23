"""Load an Arabic lexicon from a frequency-sorted word list.

The list is the CAMeL Lab's Modern Standard Arabic frequency list, fetched by
scripts/fetch_lexicon.py and never committed: it is licensed CC BY-SA 4.0 and
is 69 MB compressed.

A frequency cutoff is not optional. The list is built from web text, and web
text contains badly extracted Arabic PDFs, so it contains extraction errors
too: the garbage word this project's gate exists to catch appears in it nine
times. The cutoff is calibrated against ground truth rather than guessed, so
it has no default here.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from daleel.ingest.quality import build_lexicon

LEXICON_ZIP = Path("data/external/camel_msa_freq_lists.tsv.zip")
LEXICON_MEMBER = "MSA_freq_lists.tsv"


def read_frequency_list(lines: Iterable[str], *, min_frequency: int) -> Iterator[tuple[str, int]]:
    """Yield each word and its frequency, for words seen at least `min_frequency` times.

    The list is sorted by descending frequency, so reading stops at the first
    word below the cutoff. That shortcut would silently drop words from an
    unsorted list, so the order is checked as it is read, and any malformed
    line is an error rather than something to skip.
    """
    previous = None
    for number, line in enumerate(lines, start=1):
        word, tab, count = line.rstrip("\r\n").partition("\t")
        if not tab or not word or not count.isdigit():
            raise ValueError(f"line {number} is not 'word<TAB>frequency': {line!r}")
        frequency = int(count)
        if previous is not None and frequency > previous:
            raise ValueError(f"line {number}: frequency rises from {previous} to {frequency}")
        if frequency < min_frequency:
            return
        previous = frequency
        yield word, frequency


def load_lexicon(
    zip_path: Path = LEXICON_ZIP, *, min_frequency: int, member: str = LEXICON_MEMBER
) -> frozenset[str]:
    """Read the word list straight from its zip, cut it, and normalize it."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as raw:
        lines = io.TextIOWrapper(raw, encoding="utf-8")
        entries = read_frequency_list(lines, min_frequency=min_frequency)
        return build_lexicon(word for word, _ in entries)
