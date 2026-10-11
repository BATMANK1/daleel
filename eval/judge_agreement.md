# Judge agreement (T9)

The judge, Gemma 4 31B with rubric 2, gives the same label as a person on 37
of 40 answers: Cohen's kappa 0.89, with a 95% bootstrap interval of 0.76 to
1.00. Rubric 2 was written after reading where rubric 1 disagreed on these
same 40 answers, so that figure is optimistic. Rubric 1, written before any
answer was labelled, gave kappa 0.65 (0.43 to 0.84) on 37. A check on
answers that played no part in writing the rubric is still to come (last
section).

## Setup

- **Answers:** 40 dev answers drawn by `scripts/draw_labels.py`, 20 from arm
  A (no documents) and 20 from the answering flow. Every dev question whose
  correct answer is not a plain answer is in each arm's 20: a refusal, a
  named gap, both sources, or the cases told apart, 9 per arm. The order is
  shuffled and the arm is not shown.
- **Hand labels:** made on 11 October 2026 by the author, from the sheet the
  script writes, before any label from the judge was seen. The sheet shows
  the definitions in `src/daleel/eval/judge.py` and, for each answer, the
  same six parts the judge reads: the question, what a correct answer does,
  the reference answer, the evidence quotes, the sources the answer cites,
  and the answer. The hand labels were not changed after the judge's were
  read.
- **Judge:** `gemma-4-31b-it` through the Gemini API's free tier, at the
  model's default temperature, with up to 8,192 tokens per reply, run by
  `scripts/judge_agreement.py`.
- **Hand labels given:** 13 correct, 12 partial, 12 incorrect, 3
  hallucinated. All 12 incorrect are arm A's declines: of questions the
  documents answer, and in g073's case answer in part.

## Rubric 1

| 37 answers | judge: correct | judge: partial | judge: incorrect | judge: hallucinated |
|---|---|---|---|---|
| human: correct | 9 | 2 | 0 | 0 |
| human: partial | 0 | 7 | 4 | 0 |
| human: incorrect | 0 | 1 | 11 | 0 |
| human: hallucinated | 0 | 1 | 1 | 1 |

Kappa 0.65; agreement on 28 of 37. Three answers got no verdict: at
temperature 0 the judge spent all of its 2,048 tokens thinking.

The nine disagreements fell into four kinds:

| Kind | Answers | Hand | Judge |
|---|---|---|---|
| A decline of a question the documents answer in part | 4 | partial | incorrect |
| A correct refusal that leaves out a side remark of the reference | 2 | correct | partial |
| Steps in the wrong order; a condition invented before a decline | 2 | hallucinated | partial, incorrect |
| A decline where the reference itself is incomplete (g073) | 1 | incorrect | partial |

The first three are rules the person labelling applied and the rubric did
not state.

## Rubric 2

Rubric 2 states them: a decline makes only the points of the reference that
say the documents lack something; for a question the documents do not
answer at all, refusing is the one point that counts; and steps out of
order, or a detail stated before a decline, are hallucinated.

| 40 answers | judge: correct | judge: partial | judge: incorrect | judge: hallucinated |
|---|---|---|---|---|
| human: correct | 13 | 0 | 0 | 0 |
| human: partial | 2 | 10 | 0 | 0 |
| human: incorrect | 1 | 0 | 11 | 0 |
| human: hallucinated | 0 | 0 | 0 | 3 |

Kappa 0.89; agreement on 37 of 40, 18 of 20 for arm A and 19 of 20 for the
flow. Every answer got a verdict.

All three disagreements have the judge more lenient than the hand label, and
each credits a decline:

- **g072, arm A** (hand partial, judge correct). The documents give the
  general rules for changing major but not which majors take students in
  term 481. The judge treated the question as one the documents do not
  answer at all, so the decline made the one point that counts. Rubric 2's
  rule for refusals reached a question it was not written for.
- **g073, both arms** (hand partial and incorrect, judge correct). The
  reference answer says the documents do not explain how to cancel a change
  of major. The regulations do give one rule: the change may be reversed
  until the end of the add and drop period at the start of the term in
  which it was approved (organizational regulations, page 18). The hand labels used it; the judge
  followed the reference. g073 joins the candidates for gold_v2.

So the judge may still credit a decline on a question the documents answer
in part. That favors arm A, which declines most, and the arms' accuracy
should be read with it in mind.

## Running the judge

At temperature 0 the judge's thinking ran on until it hit the reply's limit:
3 of 40 answers had no verdict at 2,048 tokens, and at 8,192 the API failed
with errors 500 and 503 for an hour. At the default temperature the 40
answers took about an hour, through four runs, because the free tier's
Gemma 31B often answers 500 or 503 ("high demand"). The client asks again
after a pause, and a run that stops keeps every verdict it got.

## Still to do

Check rubric 2 on answers that played no part in writing it: 20 answers of
arm B, drawn the same way once arm B is run, labelled by hand before the
judge sees them.
