"""Hybrid retrieval: rankings from several retrievers, fused, then reranked.

Each retriever ranks the chunks for a question down to CANDIDATES deep. One
ranking is used as it is, several are fused by reciprocal rank fusion
(daleel.retrieve.fusion), and a cross-encoder, where one is given, reorders
the head of the result (daleel.retrieve.rerank).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from daleel.retrieve.fusion import rrf
from daleel.retrieve.rerank import RERANK_DEPTH, Scorer, rerank

# How deep each retriever ranks before fusion.
CANDIDATES = 100

Search = Callable[[str, int], Sequence[str]]


def fusions(bm25: Search, dense: Mapping[str, Search]) -> dict[str, list[Search]]:
    """The fusions compared: BM25 with each dense encoder, and with all of them."""
    found = {f"BM25 + {name}": [bm25, search] for name, search in dense.items()}
    if len(dense) > 1:
        found["BM25 + " + " + ".join(dense)] = [bm25, *dense.values()]
    return found


@dataclass
class Hybrid:
    """Retrievers fused, and their head reranked where a scorer is given."""

    retrievers: Sequence[Search]
    scorer: Scorer | None = None
    texts: Mapping[str, str] | None = None
    depth: int = CANDIDATES
    rerank_depth: int = RERANK_DEPTH

    def __post_init__(self) -> None:
        if not self.retrievers:
            raise ValueError("a hybrid needs at least one retriever")
        if self.scorer is not None and self.texts is None:
            raise ValueError("reranking needs the chunks' texts")

    def search(self, question: str, k: int) -> list[str]:
        """The ids of the k best chunks for a question, best first."""
        rankings = [list(search(question, self.depth)) for search in self.retrievers]
        ranked = rankings[0] if len(rankings) == 1 else rrf(rankings)
        if self.scorer is not None and self.texts is not None:
            ranked = rerank(question, ranked, self.texts, self.scorer, self.rerank_depth)
        return ranked[:k]
