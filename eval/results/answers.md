# Answers on the development questions (T5, T7 and T8)

The judge labels the answering flow's answer correct on 0.77 of the 60
development questions, with a 95% interval of 0.65 to 0.87. With no documents,
the same model scores 0.15. With the whole corpus in its context (arm B), the
model scores 0.83. That is 0.07 above the flow, but the interval of the
difference, -0.03 to +0.17, includes no difference. Arm B also costs about 50
times as much, and its 95th-percentile time is 51.5 seconds against the flow's
2.7. So the flow stays the pipeline. Most of the questions arm B wins are
questions in several parts whose evidence retrieval ranks below tenth, and no
retry by the flow reaches that far.

These are development numbers. Arm C's prompt and the flow's rules were set
after reading answers to these questions. The numbers reported for the project
come from one run of the held-out questions, with the pipeline frozen
([held_out.md](held_out.md)).

## Setup

- **Questions:** the 60 development questions of
  [`eval/gold_v1.jsonl`](../gold_v1.jsonl).
  - 51 want a plain answer.
  - 4 want an answer that names what the documents leave out.
  - 2 want a refusal, because the corpus does not hold the answer.
  - 1 wants both sources, because two documents disagree.
  - 2 want the cases told apart.
- **Arms:** all use `gemini-3.5-flash-lite` at its default temperature
  (`src/daleel/answer/arms.py`).
  - *A:* the question alone, with no documents.
  - *B:* arm C's prompt, with all 768 chunks as sources. That is about
    140,000 tokens per question.
  - *C:* arm C's prompt, with the first five chunks of T4's chosen retrieval
    as sources.
  - *flow:* the answering flow (`src/daleel/answer/flow.py`).
    - It refuses without asking the model when the reranker's best score is
      below -4.0 ([refusal.md](refusal.md)).
    - Otherwise it asks with arm C's prompt and five sources.
    - It asks once more, with ten sources, when the answer declines, cites
      nothing, or cites a source it was not given.
    - If the second answer also declines, that decline stands. A second
      answer that fails in any other way is refused.
    - It does not rewrite the question ([rewrite.md](rewrite.md)).
- **Judge:** `gemma-4-31b-it` with rubric 2. It agrees with hand labels on 56
  of 60 answers, kappa 0.90 ([judge_agreement.md](../judge_agreement.md)).
  Accuracy is the share of answers it labels correct.
- **Measures that need no judge:** `scripts/results.py` works these out again
  from each answer.
  - *Numbers exact:* the numeric questions whose answer states the gold
    number.
  - *Citation precision:* the share of the chunks an answer cites that hold a
    gold quote, averaged over the answers that cite.
- **Intervals:** 95% bootstrap intervals, from 10,000 resamples of the
  questions. A difference between two arms is resampled on the same
  questions for both.
- **Runs:**
  - The answers were run from 9 to 12 October 2026, once per arm, on the
    Gemini API's free tier.
  - Seconds count only the request that succeeded. Waits for the free tier's
    limits and retried errors are left out. For the flow, the seconds of its
    requests are added up.
  - Dollars are at paid rates: $0.30 per million tokens in and $2.50 per
    million out, with thinking counted as output.

## T5: quality

| run | n | accuracy | 95% interval | correct or partial | numbers exact | citation precision | hallucinated | hallucinated, outside corpus | accuracy minus flow |
|---|---|---|---|---|---|---|---|---|---|
| A | 60 | 0.15 | 0.07 to 0.25 | 14/60 | 1/16 | - | 12/60 | 0/2 | -0.62 (-0.75 to -0.48) |
| B | 60 | 0.83 | 0.73 to 0.92 | 57/60 | 16/16 | 0.73 | 2/60 | 0/2 | +0.07 (-0.03 to +0.17) |
| C | 60 | 0.75 | 0.63 to 0.85 | 57/60 | 16/16 | 0.81 | 1/60 | 0/2 | -0.02 (-0.05 to +0.00) |
| flow | 60 | 0.77 | 0.65 to 0.87 | 59/60 | 16/16 | 0.82 | 1/60 | 0/2 | - |

## T7: time and cost

| run | p50 s | p95 s | tokens in | tokens out | $ per 1,000 questions |
|---|---|---|---|---|---|
| A | 1.1 | 7.0 | 163 | 39 | 0.15 |
| B | 10.7 | 51.5 | 139,910 | 96 | 42.21 |
| C | 1.2 | 2.0 | 1,521 | 123 | 0.76 |
| flow | 1.3 | 2.7 | 1,714 | 128 | 0.83 |

The times are the free tier's, and they vary. Arm A's 95th percentile comes
from short declines that took 7 to 10 seconds. Retrieval time is not
included; T4 gives it.

## T8: correct answers by question type

| type | n | A | B | C | flow |
|---|---|---|---|---|---|
| single_clause | 19 | 3 | 16 | 16 | 17 |
| numeric | 11 | 1 | 10 | 11 | 11 |
| multi_clause | 7 | 0 | 6 | 2 | 2 |
| table_lookup | 6 | 0 | 6 | 6 | 6 |
| out_of_scope | 6 | 4 | 5 | 3 | 3 |
| cross_document | 5 | 0 | 4 | 3 | 3 |
| precedence | 3 | 0 | 1 | 1 | 1 |
| english_query | 3 | 1 | 2 | 3 | 3 |

The out_of_scope type holds the 2 questions to refuse and the 4 that want a
gap named. Arm A's 4 there are declines. On questions the documents answer in
part, the judge can credit a decline ([judge_agreement.md](../judge_agreement.md)).

## Arm B against the flow

The two arms give the same label on 48 of the 60 questions.

- **Arm B is correct and the flow is not on 7.**
  - Four are questions in several parts where retrieval misses part of the
    evidence: g041, g042, g045 and g059.
  - One, g044, is a part the flow left out although its sources held it.
  - Two want a gap named: g069 and g072.
- **The flow is correct and arm B is not on 3.**
  - On g032, arm B gives the student guide's 20% absence limit, but says a
    student is barred on reaching 20% where the guide says on exceeding it.
    Among its four citations are the guidance manual's 15% rule and a
    paragraph on managing time.
  - On g078, arm B says a book is lost a month after the library asks for it.
    It takes this from a clause about suspending borrowing.
  - On g004, arm B names the menu but leaves out the steps that follow.

Arm B's two hallucinations are what more context costs. With every document
in front of it, the model draws on clauses from other documents and other
rules, and its citation precision falls from 0.82 to 0.73.

## The flow against arm C

The flow's first request is arm C's request, so the two arms give the same
answer on 54 of the 60 questions. On the other six, the flow either asked
again or refused:

| question | arm C | flow | labels, C and flow |
|---|---|---|---|
| g008 | declines, naming the guide the question names | asks again with ten sources, and answers from the general guide | incorrect, correct |
| g015 | cites a document by name instead of by number | asks again and cites by number | correct, correct |
| g065 | declines | asks again and answers in part | incorrect, partial |
| g071 | declines, naming the missing figure | asks again, and declines without naming it | partial, partial |
| g075 | declines, naming what is missing | asks again, and declines in other words | correct, correct |
| g076 | declines, naming what is missing | refuses without asking the model | correct, correct |

So the retry gains one correct answer on dev (g008) and turns one incorrect
answer into a partial one (g065). It costs 13% more tokens, and 0.7 seconds
more at the 95th percentile. The flow is kept for what it guarantees more
than for its accuracy:

- every answer it returns, other than a decline, cites a source it was
  given;
- a question that retrieval finds nothing close to is refused before the
  model sees it.

## Where the flow falls short

The flow's answer is not labelled correct on 14 questions. 11 of them are
questions the documents answer.

- **Retrieval, 6:** g041, g042, g045, g046, g059 and g066 lack part of their
  evidence among their five sources. On three of them (g042, g046 and g059),
  the answer says the documents do not mention a part that they do mention.
- **Generation, 5:** these have all their evidence among their sources.
  - Three leave out one clause the sources give: g009, the 30% limit; g019,
    the two weeks allowed for an appeal; and g044, the fine of twice a lost
    book's price.
  - g063 gives a procedure's steps out of order. It is the flow's one
    hallucination.
  - g065 does not match the loose document names in the question to the
    formal titles in its sources. Arm B fails it too.

The other 3 are questions the documents answer in part (g069, g071 and g072).
Each answer names the gap but leaves out a point of the reference.

## Decisions

- **The flow, not arm B, answers questions.**
  - Arm B costs about 50 times as much per question.
  - Its 95th-percentile time is 51.5 seconds.
  - The free tier allows 250,000 tokens a minute, which is fewer than two of
    arm B's questions a minute.
  - Arm B works only while the whole corpus fits in the model's context.
  - Its gain over the flow is not resolved by 60 questions.
- **No wider retry for an answer that names a gap.** For five of the six
  retrieval misses, the missing evidence is ranked below tenth, so asking
  again with ten sources would not find it. The exception is g066, whose last
  quote is seventh.
- **The flow keeps the second decline when both of its tries decline.**
  Keeping the first instead would change no label on dev: g071 is partial
  either way.

Not tried: adding each source's neighbouring chunks. With the chunk before and
the chunk after each of the five sources, all of the evidence for g041, g042
and g066 would be present. But it would about triple the tokens, and a gain
of at most three questions (0.05) is smaller than 60 questions can tell from
chance. It is left for after the held-out run.

## Limits

- **Few questions:** one arm's interval is about 0.2 wide. Paired
  differences are narrower, but two arms that differ by a few questions are
  not told apart.
- **One run per arm:** the model samples, so a second run would differ on
  some answers. The replies are stored, so every table here can be rebuilt
  from the store without calling the API.
- **The judge:** it may credit a decline of a question the documents answer
  in part. That favors the arms that decline most. On dev it moves an arm's
  accuracy by at most 0.07 ([judge_agreement.md](../judge_agreement.md)).
