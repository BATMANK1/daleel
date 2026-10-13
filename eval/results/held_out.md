# Answers on the held-out questions (T5, T7 and T8)

These are the numbers reported for the project. The pipeline was frozen at
the tag `frozen-v1` before any held-out question was answered, and each arm
answered the 20 held-out questions once.

The judge labels the answering flow's answer correct on 13 of the 20
questions: 0.65, with a 95% interval of 0.45 to 0.85. The flow's answer is
correct or partial on 19. With the whole corpus in its context, arm B also
scores 0.65, at about 44 times the flow's cost. With no documents, the model
scores 0.10. Arm C, which is the flow without its checks, scores 0.70. The
0.05 between arm C and the flow is one question, g070. There the two answers
say the same thing in almost the same words, and the judge labels them
differently.

Twenty questions cannot tell arms B, C and the flow apart. Each of their
intervals is 0.4 wide, and the interval of the difference between any two of
them includes no difference.

## Setup

- **Questions:** the 20 held-out questions of
  [`eval/gold_v1.jsonl`](../gold_v1.jsonl).
  - 17 want a plain answer.
  - 1 wants an answer that names what the documents leave out (g070).
  - 1 wants a refusal, because the corpus does not hold the answer (g074).
  - 1 wants both sources, because two documents disagree (g067).
  - 19 are in Arabic and 1 in English (g077).
- **Arms, judge, measures and intervals:** as on the development questions
  ([answers.md](answers.md#setup)), at the tag `frozen-v1`.
  - The flow has one addition since that write-up: a question about the
    current date or week gets a source computed from the academic calendar
    (`src/daleel/answer/today.py`). No held-out question asks about time, so
    it played no part here.
- **Runs:**
  - The answers were run on 13 October 2026, once per arm, on the Gemini
    API's free tier.
  - The flow's first request is arm C's request. On 17 questions the flow
    asked once, so its reply came from the store, with the seconds that arm
    C's request took. On the other 3, the seconds of both its requests are
    added up.
- **After the run:** no prompt, rule or setting was changed. One measure
  that needs no judge was fixed; see [the number
  measure](#a-fix-to-the-number-measure).

## T5: quality

| run | n | accuracy | 95% interval | correct or partial | numbers exact | citation precision | hallucinated | hallucinated, outside corpus | accuracy minus flow |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.10 | 0.00 to 0.25 | 3/20 | 1/6 | - | 5/20 | 0/1 | -0.55 (-0.75 to -0.35) |
| B | 20 | 0.65 | 0.45 to 0.85 | 18/20 | 6/6 | 0.78 | 2/20 | 0/1 | +0.00 (-0.20 to +0.20) |
| C | 20 | 0.70 | 0.50 to 0.90 | 19/20 | 6/6 | 0.77 | 1/20 | 0/1 | +0.05 (+0.00 to +0.15) |
| flow | 20 | 0.65 | 0.45 to 0.85 | 19/20 | 6/6 | 0.78 | 1/20 | 0/1 | - |

Arm C minus arm B is +0.05, with an interval of -0.15 to +0.25.

## T7: time and cost

| run | p50 s | p95 s | tokens in | tokens out | $ per 1,000 questions |
|---|---|---|---|---|---|
| A | 1.0 | 1.4 | 163 | 33 | 0.13 |
| B | 36.0 | 69.0 | 139,910 | 115 | 42.26 |
| C | 7.9 | 11.6 | 1,669 | 117 | 0.79 |
| flow | 7.9 | 11.6 | 2,092 | 128 | 0.95 |

The free tier was slow when these answers were run. Arm C's requests were
about the size they were on dev, 1,669 tokens in against 1,521, but its
median took 7.9 seconds against 1.2. The flow's three new requests, made
later the same day with ten sources each, took 1.1 to 1.6 seconds. So these
seconds measure the service on that day more than the model, and dev's
times are the better guide to its speed. On both days arm B is the slowest
by several times. It costs about 44 times as much as the flow here, and
about 50 times as much on dev.

## T8: correct answers by question type

| type | n | A | B | C | flow |
|---|---|---|---|---|---|
| single_clause | 6 | 0 | 5 | 5 | 5 |
| numeric | 4 | 0 | 3 | 3 | 3 |
| multi_clause | 3 | 0 | 1 | 1 | 1 |
| table_lookup | 2 | 1 | 2 | 2 | 2 |
| out_of_scope | 2 | 1 | 1 | 2 | 1 |
| cross_document | 1 | 0 | 1 | 0 | 0 |
| precedence | 1 | 0 | 0 | 1 | 1 |
| english_query | 1 | 0 | 0 | 0 | 0 |

The out_of_scope type holds g074, to refuse, and g070, which wants a gap
named. Arm A's two correct answers are its decline of g074, and g054, a
grade's range and points, which the model knew without documents.

## The flow against arm C

The two arms give the same answer on 17 of the 20 questions. On the other
three, the flow asked again with ten sources:

| question | arm C | flow | labels, C and flow |
|---|---|---|---|
| g050 | cites the student guide by name instead of by number | asks again and cites by number | partial, partial |
| g070 | declines, naming the two things the documents leave out | asks again, and declines naming the same two | correct, partial |
| g074 | declines, naming what is missing | asks again, and declines without naming it | correct, correct |

The two answers to g070 differ by two words:

- arm C: لا تذكر الوثائق المتاحة الجهة التي يُرفع إليها العذر الطبي ولا رابط أو مسار رفعه المعتمد للفصل 481.
- flow: لا تذكر الوثائق المتاحة الجهة المحددة التي يُرفع لها العذر الطبي ولا رابط أو مسار رفعه المعتمد للفصل 481.

For arm C's answer, the judge drew two points from the reference. For the
flow's, it drew five, and found three of them missing: the excuses the rules
accept, the week allowed for handing one in, and the advice to follow the
college's current instructions. Neither answer gives those, so partial is the
better label for both. The labels are reported as the judge gave them.

The flow's confidence gate refused none of the 20 questions. On g074, which
the corpus does not answer, the model declined both times.

So on these questions the retry put one answer's citations right (g050), and
changed what an answer says once, on g074, where the second decline lost the
gap the first one named. Dev had the same case in g071. The flow is kept for
what it guarantees, as decided on dev: every answer it returns, other than a
decline, cites a source it was given, and a question that retrieval finds
nothing close to is refused before the model sees it. It costs a fifth more
per question than arm C.

## Arm B against the flow

The two arms give the same label on 14 of the 20 questions.

- **Arm B is correct and the flow is not on 2.**
  - g061, a retrieval miss (below).
  - g011: arm B gives the steps for signing in through the national access
    service. The flow names the option and leaves out the two steps after it.
- **The flow is correct and arm B is not on 2.**
  - g067 asks to compare two documents that disagree, without choosing one
    unless the documents give grounds. Arm B answers yes, which takes one
    side. It also says the regulations' article is repeated on a page of the
    student guide that does not hold it. The judge labels it hallucinated.
  - g025: the two answers say much the same thing. The judge drew one more
    point from the reference for arm B's answer than for the flow's, and
    found it missing.
- **They also differ on 2 where neither is correct.**
  - g070: arm B names an office for medical excuses, from a clause about
    excuses for missed exams. The question is about a missed lecture.
    Hallucinated.
  - g047: arm B reads the preparatory-year rule the right way round, and is
    partial. The flow's answer is hallucinated (below).

Arm B's hallucination on g070 is of the kind seen on dev: with every
document in front of it, the model applies a rule written for another case.
On g067 it settles a conflict that the question asks it to show.

## Where the flow falls short

The flow's answer is not labelled correct on 7 questions. All 7 are
questions the documents answer, in whole or in part.

- **Retrieval, 1:** in g061, the student portal guide's steps for asking for
  an extra chance run from page 37 onto page 38. Only page 37 is among the
  flow's sources. The answer stops where that page does, before saving the
  request and following its status. Arm B, with every page, is correct.
- **Retrieval and generation, 1:** g047.
  - The rules for matching courses after a transfer, on page 31 of the
    regulations, are not among the five sources. So the answer says the
    documents do not say whether the matching is automatic. They say the
    department does it.
  - The answer also inverts a double negative. The student guide says
    transfer is not accepted from students who have not passed the
    preparatory year. The answer says the student must not have passed it.
    This is the flow's one hallucination, and arm C's.
- **Generation, 4:** the evidence is among the sources, and the answer gives
  only part of it.
  - g011 leaves out two steps of signing in, as above.
  - g027 and g077 are the same question in Arabic and in English. Both
    answers give the minimum average, 3.75 out of 4, and leave out the two
    other conditions that come with it. The answer to g077 is also in
    Arabic, although the prompt says to answer in the question's language.
    The rubric does not check the language, so the label does not show
    this. The three English questions on dev were answered in English.
  - g050 leaves out the rule that a training placement is not changed once
    it is assigned.
- **The judge, 1:** g070, above.

On dev, retrieval accounted for 6 of the flow's 14 misses. Here the most
common miss is an answer that stops short with all its evidence in hand,
as on dev's g009, g019 and g044.

## A fix to the number measure

The measure of numbers stated exactly was fixed after these answers were
read. The regulations' grade table prints points with a decimal comma in
Arabic-Indic digits, ٣,٥ for 3.5. Arm C's and the flow's answers to g054
and g055 copy the points that way, as the prompt asks. The measure read
٣,٥ as two numbers, 3 and 5, and counted both answers as missing the gold
number.

`src/daleel/answer/measure.py` now reads a comma between digits with one or
two digits after it as a decimal comma. Three digits after it are still a
thousands separator. The fix moves arm C and the flow from 4/6 to 6/6
numbers exact. It changes no judge label, nothing the flow does, and no
figure on dev.

## Dev and held-out

| run | dev accuracy | held-out accuracy | held-out interval |
|---|---|---|---|
| A | 0.15 | 0.10 | 0.00 to 0.25 |
| B | 0.83 | 0.65 | 0.45 to 0.85 |
| C | 0.75 | 0.70 | 0.50 to 0.90 |
| flow | 0.77 | 0.65 | 0.45 to 0.85 |

Every held-out figure is lower than dev's. That is expected: the prompt and
the flow's rules were set on dev's questions. Each arm's held-out interval
includes its dev figure.

## What this changes

Nothing in the pipeline. The decisions made on dev stand:

- **The flow, not arm B, answers questions.** It is as accurate here, at
  about a 44th of the cost.
- **The flow is kept over arm C** for its guarantees. Its accuracy here
  differs from arm C's by one judge label.

Three changes are left for later work. They must be tried on new questions,
because these 20 have now been seen:

- **Keep the first decline when both tries decline.** The second decline
  lost a named gap on g071 (dev) and g074 (here).
- **Add each source's neighbouring chunks.** g061's last steps are in the
  chunk after its source. On dev, the chunks next to the sources held the
  missing evidence of g041, g042 and g066.
- **Check the answer's language.** An English question answered in Arabic
  should be asked again.

## Limits

- **Few questions:** one question moves an arm's accuracy by 0.05.
- **The judge:** on two pairs of answers that say the same thing, g070 for
  arm C and the flow and g025 for arm C and arm B, it drew different points
  from the reference and gave different labels. On dev it agreed with hand
  labels on 56 of 60 answers ([judge_agreement.md](../judge_agreement.md)),
  so a label or two per arm on 20 questions is within what it was measured to
  do.
- **Language:** the rubric does not check the answer's language.
- **One run per arm:** the replies are stored, so these tables can be
  rebuilt without calling the API:

  ```bash
  python3 scripts/results.py --runs data/interim/runs/A-final-judged.jsonl \
      data/interim/runs/B-final-judged.jsonl data/interim/runs/C-final-judged.jsonl \
      data/interim/runs/flow-final-judged.jsonl
  ```
