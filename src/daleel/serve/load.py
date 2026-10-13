"""The service with what it answers from: the chunks, the two models and the model client.

Every question is encoded by bge-m3 and reranked by the cross-encoder as it
comes, so only the chunks' vectors are read from disk.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from daleel.answer.llm import Client
from daleel.answer.today import calendar_source
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.retrieve.dense import DENSE, ENCODERS, QUESTION, encode, load_model
from daleel.retrieve.hybrid import chosen_retrieval
from daleel.retrieve.rerank import REFUSE_BELOW, best_score, load_reranker, predict
from daleel.serve import QUERIES, REPLIES
from daleel.serve.app import create_app


def load_app(
    chunks: Path = CHUNKS,
    dense: Path = DENSE,
    device: str | None = None,
    log: Path | None = QUERIES,
    replies: Path = REPLIES,
) -> FastAPI:
    """The service over the chunks, with T4's chosen retrieval, the confidence gate and
    the answering flow, its models on `device` (the GPU where there is one)."""
    found = read_chunks(chunks)
    encoder = ENCODERS["bge-m3"]
    model = load_model(encoder, device)
    reranker = load_reranker(device)
    retrieval = chosen_retrieval(
        found,
        dense,
        None,
        encode_question=lambda question: encode(model, encoder, [question], QUESTION)[0],
        score=lambda question, texts: predict(reranker, question, texts),
    )
    return create_app(
        retrieval.search,
        found,
        Client(replies).ask,
        confidence=best_score(retrieval.scorer),
        threshold=REFUSE_BELOW,
        log=log,
        # Questions about time get the calendar's weeks, opening with today in Riyadh.
        when=calendar_source(found),
    )
