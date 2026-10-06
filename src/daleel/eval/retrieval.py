"""Retrieval measured against the gold set's evidence.

The gold set names no chunks. Each answering page carries quotes, and a chunk
holds a quote when the quote, normalized for comparison, is in the chunk's
text; a quote given as a list holds only where all its parts are in one
chunk. So the evidence a question needs is found anew for every way of
chunking the corpus, and the frozen set measures them all alike.

A question needs every quote on every page that answers it: a question on two
documents needs both. Of a ranking, the measures are:

- recall at k: the share of a question's quotes held by a chunk among the
  first k, averaged over the questions;
- complete at k: the share of questions all of whose quotes are held among
  the first k, which is what an answer drawing on several clauses needs;
- reciprocal rank at 10: one over the rank of the first chunk holding any
  of the question's quotes, or nothing where none is among the first 10,
  averaged over the questions.

Only the questions the corpus answers are measured. The rest have nothing to
find, and how a system declines them is measured on its answers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from daleel.eval.gold import ANSWERABLE
from daleel.normalize.arabic import for_comparison

K_VALUES = (1, 5, 10, 20)


def quote_parts(quote: str | Sequence[str]) -> list[str]:
    """A quote's parts, each normalized for comparison: one, or a group's."""
    return [for_comparison(part) for part in ([quote] if isinstance(quote, str) else quote)]


def evidence(
    question: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]
) -> list[frozenset[str]]:
    """For each quote the question needs, the ids of the chunks of its page that hold it."""
    pages: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for chunk in chunks:
        pages[(chunk["doc_id"], chunk["page"])].append(
            (chunk["chunk_id"], for_comparison(chunk["text"]))
        )
    found = []
    for ref in question.get("source_refs", []):
        if ref.get("role") != "answer_evidence":
            continue
        candidates = pages.get((ref.get("doc_id"), ref.get("pdf_page")), [])
        for quote in ref.get("quotes", []):
            parts = quote_parts(quote)
            found.append(
                frozenset(
                    chunk_id for chunk_id, text in candidates if all(part in text for part in parts)
                )
            )
    return found


@dataclass(frozen=True)
class Ranked:
    """One question's evidence, and where a ranking put it."""

    qid: str
    type: str
    lang: str
    # For each quote, the rank of the first chunk holding it, from 1, or None.
    ranks: tuple[int | None, ...]
    # Quotes no chunk holds at all: a fault of the chunking, not of the ranking.
    unheld: int

    def recall(self, k: int) -> float:
        return sum(rank is not None and rank <= k for rank in self.ranks) / len(self.ranks)

    def complete(self, k: int) -> bool:
        return all(rank is not None and rank <= k for rank in self.ranks)

    def reciprocal_rank(self, k: int = 10) -> float:
        held = [rank for rank in self.ranks if rank is not None and rank <= k]
        return 1 / min(held) if held else 0.0


def rank_evidence(
    question: Mapping[str, Any], units: Sequence[frozenset[str]], ranking: Sequence[str]
) -> Ranked:
    """Where in a ranking of chunk ids each of a question's quotes is first held."""
    position = {chunk_id: place for place, chunk_id in enumerate(ranking, start=1)}
    ranks = tuple(
        min((position[chunk_id] for chunk_id in unit if chunk_id in position), default=None)
        for unit in units
    )
    return Ranked(
        question["qid"],
        question["type"],
        question["lang"],
        ranks,
        sum(not unit for unit in units),
    )


def evaluate(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    search: Callable[[str, int], Sequence[str]],
    depth: int = max(K_VALUES),
) -> list[Ranked]:
    """Every answerable question searched for, and where its evidence ranked."""
    results = []
    for question in questions:
        if question["answerability_status"] not in ANSWERABLE:
            continue
        units = evidence(question, chunks)
        if not units:
            continue
        results.append(rank_evidence(question, units, search(question["question"], depth)))
    return results


def summary(results: Sequence[Ranked], k_values: Sequence[int] = K_VALUES) -> dict[str, float]:
    """The measures over a set of questions: recall and completeness at each k, and MRR@10."""
    if not results:
        return {}
    found: dict[str, float] = {"questions": len(results)}
    for k in k_values:
        found[f"recall@{k}"] = mean(result.recall(k) for result in results)
    for k in k_values:
        found[f"complete@{k}"] = mean(result.complete(k) for result in results)
    found["mrr@10"] = mean(result.reciprocal_rank(10) for result in results)
    return found
