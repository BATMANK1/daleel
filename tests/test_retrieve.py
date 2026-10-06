"""Tests for lexical retrieval: the Arabic analyzer, BM25 and the chunk index."""

from __future__ import annotations

import math

import pytest

from daleel.retrieve.analyzer import Analyzer, light_stem
from daleel.retrieve.bm25 import BM25, WITH_HEADING, ChunkIndex

DAMMA, TATWEEL = chr(0x064F), chr(0x0640)


def test_spellings_of_one_word_become_one_term() -> None:
    analyzer = Analyzer(stopwords=False, stem=False)
    assert analyzer.terms(f"الطالب{DAMMA}") == analyzer.terms("الطالب")
    assert analyzer.terms(f"الط{TATWEEL * 3}الب") == ["الطالب"]
    assert analyzer.terms("أنظمة") == analyzer.terms("انظمه")
    assert analyzer.terms("٢٥%") == ["25"]


def test_function_words_and_colloquial_question_words_are_dropped() -> None:
    assert Analyzer(stem=False).terms("وش شروط التحويل من الكلية؟") == ["شروط", "التحويل", "الكليه"]


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("الطلاب", "طلاب"),
        ("والطلاب", "طلاب"),
        ("بالكليه", "كل"),
        ("للطالب", "طالب"),
        ("المعلومات", "معلوم"),
        ("ومن", "ومن"),
        ("وقت", "وقت"),
        # As in Lucene, a و that belongs to the word comes off too, once the word
        # has four letters: light stemming trades such errors for simplicity.
        ("وقتها", "قت"),
        ("له", "له"),
    ],
)
def test_light_stemming_takes_off_one_prefix_and_its_suffixes(word: str, stem: str) -> None:
    assert light_stem(word) == stem


def test_each_step_of_the_analyzer_can_be_turned_off() -> None:
    text = f"وش الشروط{DAMMA}"
    assert Analyzer().terms(text) == ["شروط"]
    assert Analyzer(stopwords=False).terms(text) == ["وش", "شروط"]
    assert Analyzer(stem=False).terms(text) == ["الشروط"]
    assert Analyzer(normalize=False, stopwords=False, stem=False).terms(text) == ["وش", "الشروط"]


def test_a_rarer_term_weighs_more() -> None:
    bm25 = BM25([["a", "b"], ["a", "c"], ["a", "d"]])
    assert bm25.idf["b"] > bm25.idf["a"] > 0
    assert bm25.idf["a"] == pytest.approx(math.log(1 + 0.5 / 3.5))


def test_the_best_match_comes_first_and_ties_keep_their_order() -> None:
    bm25 = BM25([["x", "y"], ["a", "b"], ["a", "b"], ["a", "q", "q", "q"]])
    assert [place for place, _ in bm25.search(["a", "b"], 3)] == [1, 2, 3]
    assert bm25.search(["absent"], 3) == []


def test_a_shorter_document_with_as_many_matches_ranks_higher() -> None:
    bm25 = BM25([["a", "x", "x", "x", "x", "x"], ["a", "y"], ["z"]])
    assert [place for place, _ in bm25.search(["a"], 2)] == [1, 0]


def test_an_empty_index_finds_nothing() -> None:
    assert BM25([]).search(["a"], 5) == []


def chunk(chunk_id: str, text: str, heading: str | None = None) -> dict:
    return {"chunk_id": chunk_id, "text": text, "section_heading": heading}


def test_the_index_returns_chunk_ids_best_first() -> None:
    chunks = [
        chunk("c1", "يجب سداد الرسوم الدراسية قبل بدء الفصل"),
        chunk("c2", "تحسب نسبة الغياب من الساعات الدراسية", "المواظبة"),
        chunk("c3", "يحرم الطالب إذا تجاوز الغياب الحد المسموح"),
    ]
    index = ChunkIndex(chunks)
    assert index.search("متى يحرم الطالب من الغياب؟", 2) == ["c3", "c2"]
    assert index.search("المواظبة", 3) == []
    assert ChunkIndex(chunks, fields=WITH_HEADING).search("المواظبة", 3) == ["c2"]
