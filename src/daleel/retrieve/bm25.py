"""BM25 over the chunks, held in memory.

Okapi BM25 with Lucene's defaults, k1 = 1.2 and b = 0.75, and Lucene's
inverse document frequency, log(1 + (N - n + 0.5) / (n + 0.5)), which stays
positive however common a term is. The corpus is a few hundred chunks, so an
inverted index in a dictionary is all it needs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from daleel.retrieve.analyzer import Analyzer

K1 = 1.2
B = 0.75


class BM25:
    """Okapi BM25 over documents given as lists of terms."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.lengths = [len(document) for document in documents]
        self.average = (sum(self.lengths) / len(documents)) if documents else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for place, document in enumerate(documents):
            for term, count in Counter(document).items():
                self.postings[term].append((place, count))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - len(found) + 0.5) / (len(found) + 0.5))
            for term, found in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        """Every document that holds a query term, and its score."""
        scores: dict[int, float] = defaultdict(float)
        for term, repeats in Counter(query).items():
            for place, count in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.lengths[place] / (self.average or 1)
                scores[place] += (
                    repeats * self.idf[term] * count * (self.k1 + 1) / (count + self.k1 * norm)
                )
        return scores

    def search(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """The k best documents, highest score first, ties in document order."""
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]


# Which of a chunk's fields its indexed text is made of.
TEXT_ONLY = ("text",)
WITH_HEADING = ("section_heading", "text")


class ChunkIndex:
    """The chunks, indexed by BM25 on the terms of the fields chosen."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        analyzer: Analyzer | None = None,
        fields: Sequence[str] = TEXT_ONLY,
    ) -> None:
        self.analyzer = analyzer or Analyzer()
        self.ids = [chunk["chunk_id"] for chunk in chunks]
        texts = ["\n".join(chunk.get(field) or "" for field in fields) for chunk in chunks]
        self.bm25 = BM25([self.analyzer.terms(text) for text in texts])

    def search(self, question: str, k: int) -> list[str]:
        """The ids of the k chunks that best match a question, best first."""
        found = self.bm25.search(self.analyzer.terms(question), k)
        return [self.ids[place] for place, _ in found]
