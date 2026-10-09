#!/usr/bin/env python3
"""Answer the gold set's questions with arms A and C, and measure what needs no judge.

    python3 scripts/run_arms.py --arms A C [--split dev] [--show 5] [options]

Arm C's sources are the first five chunks of the retrieval chosen in table T4
(BM25 + bge-m3 fused, the first 20 reranked), computed from the vectors and
reranker scores stored in data/interim/, so no GPU is needed. Each answer is
asked through daleel.answer.llm, which stores it, so a second run, or a run
stopped by the day's quota, asks only what is new. The answers go to
data/interim/runs/<arm>-<split>[-<tag>].jsonl.

Prints one row per arm:
- answered: the share of answerable questions it did not decline;
- cited: the share of its answers to them with citations, all naming a source;
- precision, recall: of its citations against the gold evidence (arm C);
- numbers: numeric questions whose number it states exactly;
- refused: outside-corpus questions it declined;
- gap named: questions the corpus answers in part, answered with the gap named;
- declined wrongly: answerable questions it declined;
- gap on answerable: answerable questions where it named something missing,
  which should be rare;
- tokens in and out, seconds at the median, and US dollars per 1,000
  questions at paid rates.
Then prints some answers to read.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from daleel.answer.arms import ARMS, SOURCES, citations, request
from daleel.answer.llm import GENERATOR, MODELS, Client, QuotaExhaustedError
from daleel.answer.measure import citation_measures, declined, names_gap, states_number
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import ANSWERABLE, GOLD_V1, SPLITS, behavior, load_gold
from daleel.eval.retrieval import by_page, evidence
from daleel.retrieve.bm25 import ChunkIndex
from daleel.retrieve.dense import DENSE, DenseIndex, VectorStore
from daleel.retrieve.hybrid import Hybrid
from daleel.retrieve.rerank import RERANK, RERANK_DEPTH, RERANKER_NAME, ScoreStore

RUNS = Path("data/interim/runs")


def chosen_retrieval(chunks: list[dict], dense: Path, rerank: Path) -> Hybrid:
    """T4's chosen configuration, from the stored vectors and scores."""
    texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    bge = DenseIndex(chunks, VectorStore.load(dense / "bge-m3.npz"))
    scores = ScoreStore.load(rerank)
    return Hybrid(
        [ChunkIndex(chunks).search, bge.search], scores.score, texts, rerank_depth=RERANK_DEPTH
    )


def answer_all(
    arm: str, questions: list[dict], chunks: list[dict], retrieval, client
) -> list[dict]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    pages = by_page(chunks)
    found = []
    for question in questions:
        sources = (
            [by_id[cid] for cid in retrieval.search(question["question"], SOURCES)]
            if arm == "C"
            else []
        )
        reply = client.ask(request(arm, question["question"], sources))
        cited = citations(reply.text, len(sources))
        units = evidence(question, pages) if question["answerability_status"] in ANSWERABLE else []
        record = {
            "qid": question["qid"],
            "arm": arm,
            "split": question["split"],
            "type": question["type"],
            "lang": question["lang"],
            "behavior": behavior(question),
            "question": question["question"],
            "answer": reply.text,
            "sources": [source["chunk_id"] for source in sources],
            "cited": list(cited.numbers),
            "cited_unknown": list(cited.unknown),
            "declined": declined(reply.text),
            "gap_named": names_gap(reply.text),
            "numeric_exact": (
                states_number(reply.text, question["answer_numeric"])
                if question.get("answer_numeric") is not None
                else None
            ),
            **(
                {"citation": citation_measures(cited, sources, units)}
                if arm == "C" and units
                else {}
            ),
            "finish": reply.finish,
            "model_version": reply.model_version,
            "tokens_in": reply.tokens_in,
            "tokens_out": reply.tokens_out + reply.tokens_thinking,
            "seconds": reply.seconds,
            "usd_at_paid_rates": reply.cost(MODELS[GENERATOR]),
            "cached": reply.cached,
        }
        found.append(record)
    return found


def run_name(arm: str, split: str, tag: str) -> str:
    return f"{arm}-{split}{'-' + tag if tag else ''}.jsonl"


def share(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}" if values else "-"


def summarize(records: list[dict]) -> dict[str, str]:
    answerable = [r for r in records if r["behavior"] not in ("refuse", "answer_and_name_the_gap")]
    answered = [r for r in answerable if not r["declined"]]
    citing = [r for r in answered if r["arm"] == "C"]
    measured = [r["citation"] for r in answerable if "citation" in r]
    numeric = [r["numeric_exact"] for r in answerable if r["numeric_exact"] is not None]
    refuse = [r["declined"] for r in records if r["behavior"] == "refuse"]
    gap = [r for r in records if r["behavior"] == "answer_and_name_the_gap"]
    costs = [r["usd_at_paid_rates"] for r in records if r["usd_at_paid_rates"] is not None]
    mean = statistics.fmean
    return {
        "answered": share([not r["declined"] for r in answerable]),
        "cited": share([bool(r["cited"]) and not r["cited_unknown"] for r in citing])
        if citing
        else "-",
        "precision": f"{mean(m['precision'] for m in measured):.2f}" if measured else "-",
        "recall": f"{mean(m['recall'] for m in measured):.2f}" if measured else "-",
        "numbers": share(numeric),
        "refused": share(refuse),
        "gap named": share([not r["declined"] and r["gap_named"] for r in gap]),
        "declined wrongly": share([r["declined"] for r in answerable]),
        "gap on answerable": share([r["gap_named"] for r in answerable]),
        "tokens in": f"{mean(r['tokens_in'] for r in records):.0f}",
        "tokens out": f"{mean(r['tokens_out'] for r in records):.0f}",
        "p50 s": f"{statistics.median(r['seconds'] for r in records):.1f}",
        "$/1k": f"{1000 * mean(costs):.2f}" if costs else "-",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer the gold questions with arms A and C.")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--show", type=int, default=5, help="answers to print for reading")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--gold", type=Path, default=GOLD_V1)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--rerank", type=Path, default=RERANK / f"{RERANKER_NAME}.json")
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--tag", default="", help="added to the answer files' names")
    args = parser.parse_args(argv)
    for path in (args.chunks, args.gold, args.dense / "bge-m3.npz", args.rerank):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    chunks = read_chunks(args.chunks)
    questions = [q for q in load_gold(args.gold) if q.get("split") == args.split][: args.limit]
    retrieval = chosen_retrieval(chunks, args.dense, args.rerank)
    client = Client()
    rows = {}
    for arm in args.arms:
        try:
            records = answer_all(arm, questions, chunks, retrieval, client)
        except QuotaExhaustedError as error:
            print(f"stopped: {error}. Answers so far are stored; run again later.", file=sys.stderr)
            return 3
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / run_name(arm, args.split, args.tag)
        target.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        rows[arm] = summarize(records)

    columns = list(next(iter(rows.values())))
    print(f"| {len(questions)} {args.split} questions, {GENERATOR} | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for arm, row in rows.items():
        print(f"| arm {arm} | " + " | ".join(row[c] for c in columns) + " |")
    print(f"\n{client.asked} requests sent now; the rest came from the store.")

    picked = random.Random(1448).sample(questions, min(args.show, len(questions)))
    for question in picked:
        print(f"\n=== {question['qid']} ({question['type']}): {question['question']}")
        for arm in args.arms:
            lines = (
                (args.out / run_name(arm, args.split, args.tag))
                .read_text(encoding="utf-8")
                .splitlines()
            )
            record = next(
                json.loads(line) for line in lines if json.loads(line)["qid"] == question["qid"]
            )
            print(f"--- arm {arm}, cites {record['cited']}:\n{record['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
