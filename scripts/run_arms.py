#!/usr/bin/env python3
"""Answer the gold set with arms A, C and flow, and measure what needs no judge.

    python3 scripts/run_arms.py --arms C flow [--split dev] [--show 5] [options]
Arm C's sources are the first five chunks of the retrieval chosen in table T4
(BM25 + bge-m3 fused, the first 20 reranked), computed from the vectors and
reranker scores stored in data/interim/, so no GPU is needed. The flow arm
uses the same retrieval, with a confidence gate, citation checks and an
optional second attempt using ten sources. Each answer is asked through
daleel.answer.llm, which stores it, so a second run, or a run stopped by the
day's quota, asks only what is new. The answers go to
data/interim/runs/<arm>-<split>[-<tag>].jsonl.
Prints one row per arm:
- answered: the share of answerable questions it did not decline, either in
  the fixed words or by saying only what the documents leave out, citing
  nothing;
- cited: the share of its answers to them with citations, all naming a source;
- precision, recall: of its citations against the gold evidence (C and flow);
- numbers: numeric questions whose number it states exactly;
- refused: outside-corpus questions it declined;
- gap named: questions the corpus answers in part, answered with the gap named;
- declined wrongly: answerable questions it declined;
- gap on answerable: answerable questions where it named something missing,
  which should be rare;
- retried: share of all questions with more than one model attempt (flow only);
- refused unasked: share of all questions refused before asking the model
  (flow only);
- tokens in and out, seconds at the median, and US dollars per 1,000
  questions at paid rates. For flow, usage covers all attempts.
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
from daleel.answer.flow import build_flow
from daleel.answer.llm import GENERATOR, MODELS, Client, QuotaExhaustedError
from daleel.answer.measure import citation_measures, declines, names_gap, states_number
from daleel.chunk.chunker import CHUNKS, read_chunks
from daleel.eval.gold import ANSWERABLE, GOLD_V1, SPLITS, behavior, load_gold
from daleel.eval.retrieval import by_page, evidence
from daleel.retrieve.dense import DENSE
from daleel.retrieve.hybrid import chosen_retrieval
from daleel.retrieve.rerank import REFUSE_BELOW, RERANK, RERANKER_NAME, best_score

RUNNABLE = (*ARMS, "flow")
RUNS = Path("data/interim/runs")


def answer_all(
    arm: str, questions: list[dict], chunks: list[dict], retrieval, client
) -> list[dict]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    pages = by_page(chunks)
    flow = (
        build_flow(
            retrieval.search,
            chunks,
            client.ask,
            confidence=best_score(retrieval.scorer),
            threshold=REFUSE_BELOW,
        )
        if arm == "flow"
        else None
    )
    found = []
    for question in questions:
        if flow is not None:
            state = flow.invoke({"question": question["question"]})
            text = state["answer"]
            sources = state["sources"]
            replies = state.get("replies", [])
            attempts = state["attempts"]
            outcome = state["outcome"]
            reason = state["reason"]
        else:
            sources = (
                [by_id[cid] for cid in retrieval.search(question["question"], SOURCES)]
                if arm == "C"
                else []
            )
            reply = client.ask(request(arm, question["question"], sources))
            text = reply.text
            replies = [reply]
            attempts = 1
            outcome = None
            reason = None

        cited = citations(text, len(sources))
        units = evidence(question, pages) if question["answerability_status"] in ANSWERABLE else []
        last_reply = replies[-1] if replies else None
        costs = [reply.cost(MODELS[GENERATOR]) for reply in replies]
        record = {
            "qid": question["qid"],
            "arm": arm,
            "split": question["split"],
            "type": question["type"],
            "lang": question["lang"],
            "behavior": behavior(question),
            "question": question["question"],
            "answer": text,
            "sources": [source["chunk_id"] for source in sources],
            "cited": list(cited.numbers),
            "cited_unknown": list(cited.unknown),
            "declined": declines(text, cited),
            "gap_named": names_gap(text),
            "numeric_exact": (
                states_number(text, question["answer_numeric"])
                if question.get("answer_numeric") is not None
                else None
            ),
            **(
                {"citation": citation_measures(cited, sources, units)}
                if arm != "A" and units
                else {}
            ),
            "attempts": attempts,
            "outcome": outcome,
            "reason": reason,
            "finish": last_reply.finish if last_reply is not None else None,
            "model_version": last_reply.model_version if last_reply is not None else None,
            "tokens_in": sum(reply.tokens_in for reply in replies),
            "tokens_out": sum(reply.tokens_out + reply.tokens_thinking for reply in replies),
            "seconds": sum(reply.seconds for reply in replies),
            "usd_at_paid_rates": (
                sum(cost for cost in costs if cost is not None)
                if all(cost is not None for cost in costs)
                else None
            ),
            "cached": all(reply.cached for reply in replies),
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
    citing = [r for r in answered if r["arm"] != "A"]
    measured = [r["citation"] for r in answerable if "citation" in r]
    numeric = [r["numeric_exact"] for r in answerable if r["numeric_exact"] is not None]
    refuse = [r["declined"] for r in records if r["behavior"] == "refuse"]
    gap = [r for r in records if r["behavior"] == "answer_and_name_the_gap"]
    costs = [r["usd_at_paid_rates"] for r in records if r["usd_at_paid_rates"] is not None]
    is_flow = bool(records) and records[0]["arm"] == "flow"
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
        "retried": share([r["attempts"] > 1 for r in records]) if is_flow else "-",
        "refused unasked": share([r["attempts"] == 0 for r in records]) if is_flow else "-",
        "tokens in": f"{mean(r['tokens_in'] for r in records):.0f}",
        "tokens out": f"{mean(r['tokens_out'] for r in records):.0f}",
        "p50 s": f"{statistics.median(r['seconds'] for r in records):.1f}",
        "$/1k": f"{1000 * mean(costs):.2f}" if costs else "-",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer gold questions with arms A, C and flow.")
    parser.add_argument("--arms", nargs="+", choices=RUNNABLE, default=list(ARMS))
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
