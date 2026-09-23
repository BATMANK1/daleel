"""Tests for reading and loading the frequency-list lexicon.

Frequencies in the load test are the real ones from the CAMeL MSA list,
including the garbage word that makes a cutoff necessary.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from daleel.ingest.lexicon import load_lexicon, read_frequency_list
from daleel.normalize.arabic import for_comparison


def line(word: str, frequency: int | str) -> str:
    return f"{word}\t{frequency}\n"


def test_words_at_or_above_the_cutoff_are_kept() -> None:
    lines = [line("في", 100), line("نظام", 50), line("وؤيتنا", 9)]
    assert list(read_frequency_list(lines, min_frequency=10)) == [("في", 100), ("نظام", 50)]


def test_reading_stops_at_the_first_word_below_the_cutoff() -> None:
    # A malformed line after the cutoff would raise if the reader reached it.
    lines = [line("في", 100), line("وؤيتنا", 9), "not a valid line"]
    assert list(read_frequency_list(lines, min_frequency=10)) == [("في", 100)]


def test_rising_frequency_is_an_error_not_a_skip() -> None:
    lines = [line("نظام", 10), line("في", 20)]
    with pytest.raises(ValueError, match="rises"):
        list(read_frequency_list(lines, min_frequency=1))


@pytest.mark.parametrize(
    "bad",
    ["no tab on this line\n", line("نظام", "many"), line("", 10), "\n"],
    ids=["no_tab", "frequency_not_a_number", "empty_word", "blank_line"],
)
def test_malformed_line_is_an_error_not_a_skip(bad: str) -> None:
    with pytest.raises(ValueError, match="not 'word<TAB>frequency'"):
        list(read_frequency_list([bad], min_frequency=1))


def test_empty_list_yields_nothing() -> None:
    assert list(read_frequency_list([], min_frequency=1)) == []


def test_lexicon_loads_from_the_zip_cut_and_normalized(tmp_path: Path) -> None:
    archive = tmp_path / "list.zip"
    rows = [line("إجازة", 146910), line("رؤيتنا", 24776), line("وؤيتنا", 9)]
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("MSA_freq_lists.tsv", "".join(rows))

    lexicon = load_lexicon(archive, min_frequency=10)

    assert for_comparison("اجازة") in lexicon  # the hamza-less spelling matches too
    assert "رؤيتنا" in lexicon
    assert "وؤيتنا" not in lexicon  # 9 occurrences on the web; the cutoff removes it
