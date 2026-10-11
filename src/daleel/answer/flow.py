"""The answering flow: a question in, a cited answer or a refusal out.

A LangGraph graph of five steps. rewrite turns the question into search
queries, the question itself unless a rewriter is given. retrieve searches
them to this attempt's depth and fuses the rankings. generate asks arm C's
prompt with those sources. verify checks the answer, cheapest check first.
An answer that fails is asked again once, with wider retrieval. After that,
a decline stands and anything else is refused with the fixed sentence.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from daleel.answer.arms import DECLINE, SOURCES, citations, request
from daleel.answer.llm import Reply, Request
from daleel.answer.measure import declines
from daleel.retrieve.fusion import rrf
from daleel.retrieve.hybrid import Search

# How many chunks each attempt retrieves: five, then ten on the retry.
DEPTHS = (SOURCES, 10)


class FlowState(TypedDict, total=False):
    """What the steps pass along. Each step returns only the fields it changes."""

    question: str
    # "ar" or "en": the language a refusal is written in.
    language: str
    queries: list[str]
    # Model calls made so far, which pick the depth of the next retrieval.
    attempts: int
    # The chunks given to the model, numbered from 1 in this order.
    sources: list[dict[str, Any]]
    # How well the sources match the question, where a confidence function is given.
    confidence: float | None
    answer: str
    cited: list[int]
    # "answered", "declined" or "refused" once the flow has decided, empty before.
    outcome: str
    # Empty, or why the last step failed: "no sources", "low confidence",
    # "declined", "no citation", "unknown citation" or "not grounded".
    reason: str
    # Every reply, for its tokens and cost. operator.add appends a step's
    # replies to the list instead of replacing it.
    replies: Annotated[list[Reply], operator.add]


def language(text: str) -> str:
    """The language to answer in: Arabic if the text has any Arabic letter, else English."""
    return "ar" if any("؀" <= char <= "ۿ" for char in text) else "en"


def build_flow(
    search: Search,
    chunks: Sequence[Mapping[str, Any]],
    ask: Callable[[Request], Reply],
    *,
    rewrite: Callable[[str], Sequence[str]] | None = None,
    confidence: Callable[[str, Sequence[Mapping[str, Any]]], float] | None = None,
    threshold: float | None = None,
    verify: Callable[[str, str, Sequence[Mapping[str, Any]]], bool] | None = None,
    depths: Sequence[int] = DEPTHS,
) -> CompiledStateGraph:
    """The answering flow, compiled. `flow.invoke({"question": q})` returns the final state.

    `search(query, k)` gives chunk ids, best first, and `ask` answers a request
    to the model. A question whose `confidence` is below `threshold` is refused
    without asking the model. `verify(question, answer, sources)`, where given,
    checks an answer that passed the other checks. Each of `depths` is one
    attempt's retrieval depth.
    """
    if threshold is not None and confidence is None:
        raise ValueError("threshold requires a confidence function")

    chunks_by_id = {chunk["chunk_id"]: dict(chunk) for chunk in chunks}

    def rewrite_node(state: FlowState) -> dict[str, Any]:
        """The queries to search, and the attempts counted from zero."""
        question = state["question"]
        return {
            "language": language(question),
            "queries": list(rewrite(question)) if rewrite is not None else [question],
            "attempts": 0,
            "replies": [],
        }

    def retrieve_node(state: FlowState) -> dict[str, Any]:
        """This attempt's sources, and whether they are too weak to answer from."""
        depth = depths[state["attempts"]]
        rankings = [list(search(query, depth)) for query in state["queries"]]
        # Several queries are fused, as T4 fuses its retrievers.
        if len(rankings) == 1:
            ids = rankings[0]
        elif rankings:
            ids = rrf(rankings)[:depth]
        else:
            ids = []

        sources = [chunks_by_id[chunk_id] for chunk_id in ids]
        score = (
            confidence(state["question"], sources) if sources and confidence is not None else None
        )

        if not sources:
            reason = "no sources"
        elif threshold is not None and score is not None and score < threshold:
            reason = "low confidence"
        else:
            reason = ""

        return {"sources": sources, "confidence": score, "reason": reason}

    def after_retrieve(state: FlowState) -> str:
        return "refuse" if state["reason"] else "generate"

    def generate_node(state: FlowState) -> dict[str, Any]:
        """Arm C's answer from this attempt's sources."""
        reply = ask(request("C", state["question"], state["sources"]))
        return {
            "answer": reply.text,
            "attempts": state["attempts"] + 1,
            "replies": [reply],
        }

    def verify_node(state: FlowState) -> dict[str, Any]:
        """Whether the answer stands, and if not, why: the cheapest checks first,
        the checker last, since it may ask a model."""
        answer = state["answer"]
        sources = state["sources"]
        found = citations(answer, len(sources))
        cited = list(found.numbers)

        if declines(answer, found):
            outcome, reason = "declined", "declined"
        elif not found.numbers and not found.unknown:
            outcome, reason = "", "no citation"
        elif found.unknown:
            outcome, reason = "", "unknown citation"
        elif verify is not None and not verify(state["question"], answer, sources):
            outcome, reason = "", "not grounded"
        else:
            outcome, reason = "answered", ""

        return {"cited": cited, "outcome": outcome, "reason": reason}

    def after_verify(state: FlowState) -> str:
        """Done, or one more attempt with wider retrieval. When none is left, a
        decline stands and anything else is refused."""
        if state["outcome"] == "answered":
            return END
        if state["attempts"] < len(depths):
            return "retrieve"
        if state["outcome"] == "declined":
            return END
        return "refuse"

    def refuse_node(state: FlowState) -> dict[str, Any]:
        """The fixed sentence that declines, in the question's language."""
        return {
            "answer": DECLINE[state["language"]],
            "cited": [],
            "outcome": "refused",
        }

    graph = StateGraph(FlowState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("refuse", refuse_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, ["generate", "refuse"])
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, ["retrieve", "refuse", END])
    graph.add_edge("refuse", END)

    return graph.compile()
