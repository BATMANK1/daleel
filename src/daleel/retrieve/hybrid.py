"""Hybrid retrieval: rankings from several retrievers, fused, then reranked.

Each retriever ranks the chunks for a question down to CANDIDATES deep. One
ranking is used as it is, several are fused by reciprocal rank fusion
(daleel.retrieve.fusion), and a cross-encoder, where one is given, reorders
the head of the result (daleel.retrieve.rerank).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daleel.retrieve.bm25 import ChunkIndex
from daleel.retrieve.dense import DenseIndex, VectorStore
from daleel.retrieve.fusion import rrf
from daleel.retrieve.rerank import RERANK_DEPTH, Scorer, ScoreStore, rerank

# How deep each retriever ranks before fusion.
CANDIDATES = 100

Search = Callable[[str, int], Sequence[str]]


def fusions[T](bm25: T, dense: Mapping[str, T]) -> dict[str, list[T]]:
    """The fusions compared, by name: BM25 with each dense encoder, and with all of them.

    `bm25` and the values of `dense` are whatever stands for each retriever: its
    search, or its name.
    """
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


def chosen_retrieval(
    chunks: Sequence[Mapping[str, Any]],
    dense: Path,
    rerank: Path | None,
    encode_question: Callable[[str], np.ndarray] | None = None,
    score: Scorer | None = None,
) -> Hybrid:
    """T4's chosen configuration, BM25 and bge-m3 fused and the first 20 reranked,
    from the stored vectors and scores.

    A question with no stored vector is encoded by `encode_question`, and one
    with a chunk to rerank that has no stored score is scored by `score`, where
    they are given: the models, for questions not asked before. With `rerank`
    None, no scores are stored, and `score` scores them all.
    """
    if rerank is None and score is None:
        raise ValueError("with no stored scores, reranking needs a score function")
    texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    bge = DenseIndex(chunks, VectorStore.load(dense / "bge-m3.npz"), encode_question)
    stored = ScoreStore.load(rerank) if rerank is not None else ScoreStore()

    def scorer(question: str, candidates: Sequence[str]) -> Sequence[float]:
        if score is not None and stored.missing(question, candidates):
            return score(question, candidates)
        return stored.score(question, candidates)

    return Hybrid([ChunkIndex(chunks).search, bge.search], scorer, texts, rerank_depth=RERANK_DEPTH)
