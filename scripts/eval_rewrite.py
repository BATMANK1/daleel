#!/usr/bin/env python3
"""Whether rewriting a question into search queries finds more of its evidence.

    python3 scripts/eval_rewrite.py [--device cuda] [--show 5] [options]

Asks the model for search queries for each answerable dev question
(daleel.answer.rewrite), then searches with the retrieval chosen in table T4
three ways: the question as asked, the model's queries alone, and the
question with the queries, their rankings fused by reciprocal rank fusion as
the answering flow fuses them. The question as asked is searched with the
stored vectors and reranker scores, so its row repeats T4's dev row. The
queries are new, so the script loads bge-m3 and the reranker to encode and
score them, which needs the optional dependencies (pip install -e ".[dense]")
and is quickest on a GPU.

Prints each way's measures, how many questions the rewrite helped and hurt at
recall@5, and what the rewrite costs in time and tokens. The rewrites go to
data/interim/runs/rewrite-dev.jsonl. Reads the dev questions only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from daleel.answer.llm import Client, QuotaExhaustedError, Reply, Request
from daleel.answer.rewrite import rewriter
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import GOLD_V1, load_gold
from daleel.eval.retrieval import Ranked, evaluate, summary
from daleel.retrieve.dense import DENSE, ENCODERS, QUESTION, encode, load_model
from daleel.retrieve.fusion import rrf
from daleel.retrieve.hybrid import chosen_retrieval
from daleel.retrieve.rerank import RERANK, RERANKER_NAME, load_reranker, predict

RUNS = Path("data/interim/runs")
COLUMNS = ("recall@5", "recall@10", "complete@5", "mrr@10")


def changed(results: Sequence[Ranked], baseline: Sequence[Ranked]) -> tuple[list[str], list[str]]:
    """The questions that gained and that lost recall@5 against the baseline."""
    pairs = list(zip(results, baseline, strict=True))
    better = [new.qid for new, old in pairs if new.recall(5) > old.recall(5)]
    worse = [new.qid for new, old in pairs if new.recall(5) < old.recall(5)]
    return better, worse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure rewriting questions on dev.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--rerank", type=Path, default=RERANK / f"{RERANKER_NAME}.json")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: the best found)")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--show", type=int, default=5, help="rewrites to print for reading")
    args = parser.parse_args(argv)
    for path in (args.chunks, args.gold, args.dense / "bge-m3.npz", args.rerank):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    chunks = read_chunks(args.chunks)
    questions = [q for q in load_gold(args.gold) if q.get("split") == "dev"]
    encoder = ENCODERS["bge-m3"]
    model = load_model(encoder, args.device)
    reranker = load_reranker(args.device)
    retrieval = chosen_retrieval(
        chunks,
        args.dense,
        args.rerank,
        encode_question=lambda text: encode(model, encoder, [text], QUESTION)[0],
        score=lambda text, candidates: predict(reranker, text, candidates),
    )
    searched: dict[tuple[str, int], list[str]] = {}

    def search(text: str, k: int) -> list[str]:
        if (text, k) not in searched:
            searched[text, k] = retrieval.search(text, k)
        return searched[text, k]

    client = Client()
    # One reply for each question rewritten, in the order of `rewrites`.
    replies: list[Reply] = []

    def ask(asked: Request) -> Reply:
        replies.append(client.ask(asked))
        return replies[-1]

    rewrite = rewriter(ask)
    rewrites: dict[str, list[str]] = {}

    def queries(question: str) -> list[str]:
        if question not in rewrites:
            rewrites[question] = rewrite(question)
        return rewrites[question]

    def fused(texts: Sequence[str], k: int) -> list[str]:
        rankings = [search(text, k) for text in texts]
        return rankings[0] if len(rankings) == 1 else rrf(rankings)[:k]

    ways: dict[str, Callable[[str, int], list[str]]] = {
        "as asked": search,
        "queries alone": lambda question, k: fused(queries(question)[1:] or [question], k),
        "as asked and queries, fused": lambda question, k: fused(queries(question), k),
    }
    try:
        results = {name: evaluate(questions, chunks, way) for name, way in ways.items()}
    except QuotaExhaustedError as error:
        print(f"stopped: {error}. Rewrites so far are stored; run again later.", file=sys.stderr)
        return 3

    baseline = results["as asked"]
    print(
        f"| {len(baseline)} answerable dev questions | "
        + " | ".join(COLUMNS)
        + " | better | worse |"
    )
    print("|---" * (len(COLUMNS) + 3) + "|")
    for name, found in results.items():
        measures = summary(found)
        better, worse = changed(found, baseline)
        cells = [f"{measures[column]:.3f}" for column in COLUMNS]
        print(f"| {name} | " + " | ".join(cells) + f" | {len(better)} | {len(worse)} |")
    better, worse = changed(results["as asked and queries, fused"], baseline)
    print(f"\nfused, better at recall@5: {', '.join(better) or 'none'}")
    print(f"fused, worse at recall@5: {', '.join(worse) or 'none'}")

    extra = [len(texts) - 1 for texts in rewrites.values()]
    print(
        f"\nqueries per question {statistics.fmean(extra):.1f}; rewrite p50 "
        f"{statistics.median(r.seconds for r in replies):.1f} s; tokens in "
        f"{statistics.fmean(r.tokens_in for r in replies):.0f}, out "
        f"{statistics.fmean(r.tokens_out + r.tokens_thinking for r in replies):.0f}; "
        f"{client.asked} requests sent now, the rest from the store"
    )

    by_question = {q["question"]: q["qid"] for q in questions}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "rewrite-dev.jsonl"
    records = [
        {
            "qid": by_question[question],
            "question": question,
            "queries": texts[1:],
            "seconds": reply.seconds,
            "cached": reply.cached,
        }
        for (question, texts), reply in zip(rewrites.items(), replies, strict=True)
    ]
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )

    recall = {name: {r.qid: r.recall(5) for r in found} for name, found in results.items()}
    for record in random.Random(1448).sample(records, min(args.show, len(records))):
        qid = record["qid"]
        print(f"\n=== {qid}: {record['question']}")
        for query in record["queries"]:
            print(f"  -> {query}")
        print(
            f"  recall@5 as asked {recall['as asked'][qid]:.2f}, "
            f"fused {recall['as asked and queries, fused'][qid]:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
