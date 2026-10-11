# Refusal threshold

The answering flow refuses a question without asking the model when the
cross-encoder's best score among the question's five sources is below
`REFUSE_BELOW` in `src/daleel/retrieve/rerank.py`. A score that low means
retrieval found nothing close to the question. This document says where the
line sits and why. The numbers come from `scripts/refusal_threshold.py`, using
the reranker scores stored for table T4, on the 60 development questions,
11 October 2026.

## Scores by expected behavior

| expected behavior | questions | lowest | median | highest |
|---|---|---|---|---|
| answer | 51 | -2.93 | 1.41 | 8.60 |
| answer_and_name_the_gap | 4 | -2.77 | -1.50 | 2.63 |
| refuse | 2 | -4.59 | -3.95 | -3.31 |
| show_both_sources | 1 | -0.47 | -0.47 | -0.47 |
| tell_the_cases_apart | 2 | 0.19 | 0.48 | 0.76 |

## What each threshold refuses

| refuse below | should answer, refused | outside corpus, refused |
|---|---|---|
| -5.0 | 0/58 | 0/2 |
| -4.5 | 0/58 | 1/2 |
| -4.0 | 0/58 | 1/2 |
| -3.5 | 0/58 | 1/2 |
| -3.0 | 0/58 | 2/2 |
| -2.5 | 3/58 | 2/2 |

## The rule

The two errors do not cost the same. A question refused wrongly costs a
student a correct answer, because the model never sees the question. A
question let through wrongly costs at most two model calls, and then the model
declines it itself, as arm C's prompt v3 did for both outside-corpus dev
questions. So the line sits well below every question that should be answered:
one logit under the lowest of them (g004, -2.93), rounded down to a half. That
gives -4.0. On dev it refuses g076 (-4.59), lets g075 (-3.31) through to the
model, and refuses none of the 58 questions that should be answered.

The data cannot support a tighter line. There are only two outside-corpus dev
questions, and the three lowest questions that should be answered score
-2.93, -2.77 and -2.56, all within 0.75 of g075. A line placed between them, at
-3.0 to -3.3, separates these 60 questions but would probably refuse
answerable questions on new data.

## The held-out questions

While this threshold was being set, the best scores of the 20 held-out
questions were printed once, by mistake, next to the dev scores. The rule
above was chosen after that, so the refusals it makes on the held-out
questions do not measure it blind. The script now reads only the dev
questions.
