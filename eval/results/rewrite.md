# Query rewriting

Rewriting a question into search queries finds less of its evidence than the
question itself, so the answering flow searches the question as asked. The
numbers come from `scripts/eval_rewrite.py`, on the 54 answerable dev
questions, 11 October 2026.

## Setup

- **Rewrite:** the generator, gemini-3.5-flash-lite, writes one to three short
  queries in Modern Standard Arabic for each question, in the formal terms of
  regulations, one for each thing asked (`src/daleel/answer/rewrite.py`). It
  wrote 2.2 queries per question on average.
- **Retrieval:** each text is searched with T4's chosen retrieval, BM25 and
  bge-m3 fused and the first 20 reranked. The question as asked uses the
  stored vectors and reranker scores. The queries are new, so they were
  encoded and scored by the same models on an RTX 3060 Ti.
- **Fused:** the question and its queries are each searched, and their
  rankings fused by reciprocal rank fusion, as the flow fuses them when given
  a rewriter.

## Results

| 54 answerable dev questions | recall@5 | recall@10 | complete@5 | mrr@10 | better | worse |
|---|---|---|---|---|---|---|
| as asked | 0.931 | 0.960 | 0.852 | 0.811 | 0 | 0 |
| queries alone | 0.843 | 0.934 | 0.759 | 0.732 | 0 | 7 |
| as asked and queries, fused | 0.881 | 0.934 | 0.815 | 0.750 | 1 | 4 |

Better and worse count the questions whose recall@5 rose or fell against the
question as asked. The first row repeats T4's dev row, so all three rows are
measured alike.

## Decision

The rule, fixed before the run: use the rewrite only if the fused row beats
the question as asked by at least 0.02 recall@5, does not lower complete@5,
and has more questions better than worse. The fused row is 0.050 lower at
recall@5 and completes 44 questions against 46. It is worse on four (g002,
g005, g041, g043) and better on one (g066). So the flow does not rewrite.

The queries alone find less than the question, and are worse on 7 questions
and better on none. Reciprocal rank fusion gives each query the same weight
as the question, so where the queries miss the evidence, they outvote it. A
rewrite would also add a model call to every question: 1.7 seconds at the
median, with 174 tokens in and 32 out.

The rewrites themselves, read one by one, are not always faithful. One
question in four parts (g041) gets three queries, and none asks for the
fourth part, the limit on requests found to be unfounded. Four rewrites
contain broken or English words, such as "للstudent" (g004) and "آخر date"
(g080). One adds a city the question does not name (g006).

Not tried: weighting the question above its queries, or rewriting only the
questions whose retrieval is weak. Either would need its own rule, set before
a new run.
