"""The web service: the answering flow and search behind HTTP, and the page.

/ask streams the flow's steps as server-sent events, then the answer once it
has passed its checks. The model's words are not streamed as they come,
because an answer is not shown before its citations are verified.

Every question and query is redacted as it arrives (daleel.serve.privacy),
and the query log keeps what was asked and answered, with the hour, and
nothing about who asked. Opened to invited users, the service answers only
requests that send the access code they were given.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, StringConstraints

from daleel.answer.arms import SOURCES
from daleel.answer.flow import build_flow, language
from daleel.answer.llm import LlmError, Reply, Request
from daleel.retrieve.hybrid import Search
from daleel.serve.privacy import redact

MAX_QUESTION = 500
MAX_RESULTS = 20
PAGE = Path(__file__).parent / "static" / "index.html"
UNAVAILABLE = {
    "ar": "تعذر الحصول على إجابة الآن. حاول مرة أخرى بعد قليل.",
    "en": "No answer can be given right now. Please try again shortly.",
}
# What a source shows of its chunk.
SOURCE_FIELDS = ("chunk_id", "doc_id", "doc_title", "page", "section_heading", "text")

Text = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION)
]


class Ask(BaseModel):
    question: Text


class Find(BaseModel):
    query: Text
    k: int = Field(default=SOURCES, ge=1, le=MAX_RESULTS)


def event(name: str, data: Mapping[str, Any]) -> str:
    """One server-sent event, its data as JSON on one line."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def shown(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {name: chunk.get(name) for name in SOURCE_FIELDS}


def tokens_of(replies: Sequence[Reply]) -> tuple[int, int]:
    """Tokens in and out of the replies, thinking counted as out."""
    return (
        sum(reply.tokens_in for reply in replies),
        sum(reply.tokens_out + reply.tokens_thinking for reply in replies),
    )


def create_app(
    search: Search,
    chunks: Sequence[Mapping[str, Any]],
    ask: Callable[[Request], Reply],
    *,
    confidence: Callable[[str, Sequence[Mapping[str, Any]]], float] | None = None,
    threshold: float | None = None,
    log: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    when: Callable[[str], Mapping[str, Any] | None] | None = None,
    code: str | None = None,
) -> FastAPI:
    """The service over the chunks. `search`, `ask`, `confidence`, `threshold` and `when`
    are as build_flow takes them. Each question is appended to `log`, where one is given.
    `clock` times each question, read once when it arrives and once when it is answered.
    Where `code` is given, /ask, /search and /metrics answer only a request that sends
    it in the X-Access-Code header."""
    flow = build_flow(search, chunks, ask, confidence=confidence, threshold=threshold, when=when)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    # The model client and the models serve one request at a time.
    busy = threading.Lock()

    # Each app counts in its own registry, so that two apps never share a count.
    registry = CollectorRegistry()
    asked = Counter("daleel_asks", "Questions, by outcome.", ["outcome"], registry=registry)
    timing = Histogram("daleel_ask_seconds", "Seconds to answer a question.", registry=registry)
    tokens = Counter(
        "daleel_model_tokens", "Model tokens, in and out.", ["direction"], registry=registry
    )
    searched = Counter("daleel_searches", "Searches.", registry=registry)

    def write(line: Mapping[str, Any]) -> None:
        if log is None:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    def answer(question: str) -> Iterator[str]:
        """The flow's steps as they end, then its answer or an error, as events."""
        started = clock()
        state: dict[str, Any] = {"language": language(question)}
        replies: list[Reply] = []
        failure = ""
        with busy:
            try:
                for update in flow.stream({"question": question}, stream_mode="updates"):
                    for step, changes in update.items():
                        changes = dict(changes or {})
                        # A step returns only its own replies, so they are added up here.
                        replies += changes.pop("replies", [])
                        state.update(changes)
                        yield event("step", {"step": step})
            except LlmError as error:
                failure = type(error).__name__
        took = clock() - started

        sources = state.get("sources", [])
        cited = [] if failure else state.get("cited", [])
        outcome = "error" if failure else state.get("outcome", "")
        tokens_in, tokens_out = tokens_of(replies)
        asked.labels(outcome).inc()
        timing.observe(took)
        tokens.labels("in").inc(tokens_in)
        tokens.labels("out").inc(tokens_out)
        write(
            {
                # The hour alone, so that a line cannot be matched to a moment.
                "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:00Z"),
                "question": question,
                "language": state["language"],
                "outcome": outcome,
                "reason": failure or state.get("reason", ""),
                "attempts": state.get("attempts", 0),
                "confidence": state.get("confidence"),
                "sources": [source["chunk_id"] for source in sources],
                "cited": [sources[n - 1]["chunk_id"] for n in cited],
                "answer": "" if failure else state.get("answer", ""),
                "seconds": round(took, 3),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }
        )

        if failure:
            yield event("error", {"message": UNAVAILABLE[state["language"]]})
            return
        yield event(
            "answer",
            {
                "answer": state["answer"],
                "outcome": outcome,
                "reason": state.get("reason", ""),
                "language": state["language"],
                # Only the sources the answer cites, under the numbers it cites them by.
                "sources": [{"n": n, **shown(sources[n - 1])} for n in cited],
            },
        )

    def allowed(x_access_code: Annotated[str | None, Header()] = None) -> None:
        """Lets a request through when no code is set, or when it sends the code."""
        if code is None:
            return
        sent = (x_access_code or "").encode()
        if not hmac.compare_digest(sent, code.encode()):
            raise HTTPException(status_code=401, detail="an access code is needed")

    guarded = [Depends(allowed)]
    # No generated API pages: the README describes the API, and the service may be
    # open to the internet.
    app = FastAPI(
        title="Daleel",
        summary="Cited answers from RCJY college regulations",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # The page's fonts, served from here so that the page loads nothing from other sites.
    app.mount("/static", StaticFiles(directory=PAGE.parent), name="static")

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "chunks": len(chunks)}

    @app.post("/search", dependencies=guarded)
    def find(body: Find) -> dict[str, Any]:
        with busy:
            ids = search(redact(body.query), body.k)
        searched.inc()
        return {
            "results": [
                {"rank": rank, **shown(by_id[chunk_id])}
                for rank, chunk_id in enumerate(ids, start=1)
            ]
        }

    @app.post("/ask", dependencies=guarded)
    def ask_question(body: Ask) -> StreamingResponse:
        return StreamingResponse(answer(redact(body.question)), media_type="text/event-stream")

    @app.get("/metrics", dependencies=guarded)
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app
