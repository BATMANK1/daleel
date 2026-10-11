"""Rewriting a student's question into search queries.

A student may ask in English, loosely, or about several things at once, and
the documents are formal Arabic. The model writes one to three queries in the
documents' terms. The question itself is searched too, first, so a rewrite
can add to what retrieval finds but never replace it.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from daleel.answer.llm import GENERATOR, Reply, Request

MAX_QUERIES = 3

SYSTEM_REWRITE = f"""You turn a student's question about the regulations of the \
colleges and institutes of the Royal Commission for Jubail and Yanbu into search \
queries. The regulations are written in Modern Standard Arabic.

Rules:
1. Write one to {MAX_QUERIES} short queries in Modern Standard Arabic, whatever \
the question's language, in the formal terms such regulations use.
2. If the question asks about several things, write one query for each.
3. Keep every name, number, date and document title the question gives.
4. Do not answer the question, and add nothing it does not ask about.
5. Reply with the queries only, one per line, without numbers or bullets."""

# A list marker a model may put before a line despite rule 5: 1. 1) - * and the
# like, but not the start of a number such as 3.75. \d matches Arabic-Indic digits too.
_MARKER = re.compile(r"^\s*(?:[-*\u2022]\s*|\d+[.)\-]\s+)")


def rewrite_request(question: str, model: str = GENERATOR) -> Request:
    """What the model is asked to rewrite one question."""
    return Request(model, f"Question: {question}", system=SYSTEM_REWRITE, purpose="rewrite")


def queries(text: str) -> list[str]:
    """The queries in a reply: one per line, list markers and quotes taken off,
    each once, at most MAX_QUERIES."""
    found: list[str] = []
    for line in text.splitlines():
        query = _MARKER.sub("", line).strip().strip("\"'«»").strip()
        if query and query not in found:
            found.append(query)
    return found[:MAX_QUERIES]


def rewriter(ask: Callable[[Request], Reply], model: str = GENERATOR) -> Callable[[str], list[str]]:
    """A rewrite for the answering flow: the question first, then the model's queries."""

    def rewrite(question: str) -> list[str]:
        reply = ask(rewrite_request(question, model))
        return [question, *(query for query in queries(reply.text) if query != question)]

    return rewrite
