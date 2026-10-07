"""Reciprocal rank fusion: several rankings made into one.

BM25 and a dense encoder score chunks on scales that cannot be compared, so
their rankings are fused by rank alone (Cormack, Clarke and Büttcher, 2009).
Each ranking gives a chunk 1 / (k + rank), the chunk's scores are summed,
and k = 60 keeps the first few ranks of any one list from deciding alone.
"""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Every chunk of the rankings, best fused first, ties in order of first appearance."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    order = {chunk_id: place for place, chunk_id in enumerate(scores)}
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], order[chunk_id]))
