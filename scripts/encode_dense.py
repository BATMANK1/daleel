#!/usr/bin/env python3
"""Encode the chunks and the gold set's questions with each dense encoder.

    python3 scripts/encode_dense.py [--encoders NAME ...] [--device cuda] [options]

For each encoder in daleel.retrieve.dense.ENCODERS, reads the chunks
(data/processed/chunks.jsonl) and every question of the gold set, encodes
whatever has no vector yet, and writes data/interim/dense/<encoder>.npz with
the model, its revision, the library versions and the device that made the
vectors. Vectors of texts no longer in the chunks or the gold set are
dropped. Needs the optional dependencies (pip install -e ".[dense]") and the
models, which are downloaded from Hugging Face on first use.

--model NAME=PATH loads an encoder from a local folder instead, as the tests
of this script do with a tiny stand-in model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.retrieve.dense import (
    DENSE,
    DOCUMENT,
    ENCODERS,
    MAX_TOKENS,
    QUESTION,
    VectorStore,
    chunk_texts,
    device_name,
    encode,
    load_model,
    model_revision,
    overlong,
    runtime_versions,
    text_key,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode chunks and questions densely.")
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODERS), default=sorted(ENCODERS))
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--out", type=Path, default=DENSE)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: the best found)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=PATH", help="load from a folder"
    )
    args = parser.parse_args(argv)
    for path in (args.chunks, args.gold):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
    paths = dict(item.split("=", 1) for item in args.model)

    texts = chunk_texts(read_chunks(args.chunks))
    questions = [question["question"] for question in load_gold(args.gold)]
    for name in args.encoders:
        encoder = ENCODERS[name]
        target = args.out / f"{name}.npz"
        old = VectorStore.load(target) if target.exists() else VectorStore()
        keep = {text_key(DOCUMENT, text) for text in texts}
        keep |= {text_key(QUESTION, text) for text in questions}
        store = VectorStore({key: vector for key, vector in old.vectors.items() if key in keep})
        todo = {
            DOCUMENT: store.missing(DOCUMENT, texts),
            QUESTION: store.missing(QUESTION, questions),
        }
        if not any(todo.values()):
            print(f"{name}: every vector is already in {target}")
            continue

        started = time.perf_counter()
        model = load_model(encoder, args.device, paths.get(name))
        loaded = time.perf_counter() - started
        seconds, cut = {}, {}
        for kind, items in todo.items():
            cut[kind] = overlong(model, encoder, items, kind)
            started = time.perf_counter()
            if items:
                store.add(kind, items, encode(model, encoder, items, kind, args.batch_size))
            seconds[kind] = round(time.perf_counter() - started, 2)
            if cut[kind]:
                print(
                    f"{name}: {cut[kind]} {kind} texts are longer than {MAX_TOKENS} tokens "
                    "and were cut",
                    file=sys.stderr,
                )
        store.meta = {
            "encoder": name,
            "model": paths.get(name) or encoder.model,
            "revision": None if name in paths else model_revision(encoder.model),
            "query_prompt": encoder.query_prompt,
            "device": device_name(model),
            "versions": runtime_versions(),
            "seconds_to_load": round(loaded, 2),
            "seconds_to_encode": seconds,
            "encoded": {kind: len(items) for kind, items in todo.items()},
            "cut_at_max_tokens": cut,
        }
        store.save(target)
        print(
            f"{name}: {len(todo[DOCUMENT])} chunks in {seconds[DOCUMENT]} s and "
            f"{len(todo[QUESTION])} questions in {seconds[QUESTION]} s on {store.meta['device']}, "
            f"written to {target}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
