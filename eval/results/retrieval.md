# Retrieval (T3 and T4)

This document measures retrieval against the gold set: how much of the
evidence each question needs is ranked among the first five chunks. Table T3
measures BM25, and how much of it the extraction and each step of the Arabic
analyzer contribute. Table T4 adds two dense encoders, their fusion with
BM25, and a cross-encoder that reranks the fused result.

T3's numbers come from `scripts/ablate_normalization.py` and
`scripts/eval_retrieval.py`, run on two machines with two versions of
poppler. T4's come from `scripts/eval_hybrid.py` and
`scripts/time_retrieval.py`, over vectors and scores computed on one GPU.
Every design choice rests on the development questions alone. The held-out
questions are reported for T3, which involves no tuning, and for T4 once its
configuration was fixed (section 5).

## Setup

- **Gold set:** [`eval/gold_v1.jsonl`](../gold_v1.jsonl), frozen on
  6 October 2026 after every answer was checked by hand against its page. Of
  its 80 questions the corpus answers 72. Of those, 54 are for development
  and 18 are held out, chosen in whole topic groups with each type in its
  share ([README](../../README.md#gold-set)). The other 8 have nothing to
  retrieve and are measured on answers instead.
- **Evidence:** the gold set names pages and quotes, not chunks. A chunk of
  the cited page holds a quote when the quote's words, normalized for
  comparison, appear in the chunk one after another in the same order.
  Punctuation and spacing do not count. A quote given as a group of parts,
  such as a table row's cells, holds only where every part is in one chunk.
  A question needs every quote on every page that answers it: 94 quotes for
  the 54 development questions and 30 for the 18 held out.
- **Measures** (`daleel.eval.retrieval`):
  - *held*: the share of the quotes that any chunk of their page holds at
    all. It caps every other measure, because a quote that extraction
    garbled cannot be found.
  - *recall@k*: the share of a question's quotes held among the first k
    chunks, averaged over questions.
  - *complete@k*: the share of questions with every quote among the first k,
    which is what an answer drawn from several clauses needs.
  - *MRR@10*: one over the rank of the first chunk holding any quote, and 0
    beyond rank 10.
- **The pipeline's corpus:** 768 chunks from `daleel chunk`, cut from the
  records `daleel extract` wrote on machine A, where 177 pages kept their
  text layer and 80 were read by OCR. They include the calendar's 37 cards
  and the fee table typed in by hand.
- **The naive corpus:** every page read by poppler's `pdftotext -enc UTF-8`,
  split at its form feeds and cut by the same chunker: no OCR, no calendar
  cards, no fee table. Using the same chunker keeps the comparison about
  extraction. A truly naive pipeline would also cut by size.
- **Retriever:** BM25 with Lucene's k1 = 1.2, b = 0.75 and inverse document
  frequency, held in memory, over each chunk's text. Adding the section
  heading to the indexed text measured slightly worse (recall@5 0.811
  against 0.825 on the draft set), so it is left out. *Terms as written*
  means words lowercased and nothing more. The analyzer's three steps are:
  - normalization for comparison;
  - a stopword list, including the colloquial question words of Saudi
    Arabic;
  - Lucene's Arabic light stemmer.
- **Dense retrieval (T4):**
  - *Encoders:* BAAI's bge-m3 and intfloat's multilingual-e5-large-instruct,
    each encoding the chunk's text as BM25 indexes it. e5 reads each question
    after an instruction, "Given a student's question about college
    regulations, retrieve the passages that answer it", and bge-m3 takes the
    question as it is. Both read at most 512 tokens, and no chunk or question
    is longer. A question's chunks are ranked by cosine similarity, exactly,
    over all 768.
  - *Fusion:* reciprocal rank fusion with k = 60 over the first 100 chunks of
    each ranking.
  - *Reranking:* BAAI's bge-reranker-v2-m3 reads the question with each of
    the first 20 fused chunks and reorders them by its score. The chunks after
    them keep their order.
  - *Stored, then measured:* `scripts/encode_dense.py` and
    `scripts/score_rerank_pool.py` ran the models once, on machine A's GPU in
    half precision, and stored every vector and score under a hash of the
    text it came from, with the model's revision and the library versions:
    bge-m3 at 5617a9f, multilingual-e5-large-instruct at 274baa4 and
    bge-reranker-v2-m3 at 953dc6f.
    Every T4 row is computed from those files, and reading them needs
    neither the models nor a GPU.
- **Machines:**
  - *A*: Windows with WSL 2, Python 3.12.14, poppler 26.01.0. For T4, an
    NVIDIA GeForce RTX 3060 Ti with 8 GB, driver 610.88, PyTorch 2.14.1 for
    CUDA 13.0, sentence-transformers 6.1.0 and transformers 5.19.0.
  - *B*: Linux, Python 3.12.3, pypdfium2 5.12.1 (PDFium 152.0.7947.0),
    poppler 24.02.0.

  Both chunk the same records, and each reads the calendar's cards from its
  own copy of the PDF. The pipeline's rows came out identical on both. Only
  pdftotext runs independently on each machine, and only its rows differ.

## T3

BM25 over the 54 development questions the corpus answers:

| | held | recall@5 | recall@10 | complete@5 | MRR@10 |
|---|---|---|---|---|---|
| pdftotext 24.02, terms as written | 0.755 | 0.154 | 0.154 | 0.148 | 0.093 |
| pdftotext 26.01, terms as written | 0.681 | 0.154 | 0.154 | 0.148 | 0.093 |
| pdftotext 24.02, full analyzer | 0.755 | 0.662 | 0.758 | 0.593 | 0.585 |
| pdftotext 26.01, full analyzer | 0.681 | 0.597 | 0.634 | 0.500 | 0.500 |
| pipeline, terms as written | 1.000 | 0.569 | 0.705 | 0.519 | 0.471 |
| + normalization | 1.000 | 0.752 | 0.827 | 0.685 | 0.576 |
| + stopwords | 1.000 | 0.758 | 0.815 | 0.685 | 0.566 |
| **+ light stemming** | **1.000** | **0.841** | **0.897** | **0.759** | **0.681** |

The same configurations over the 18 held-out questions:

| | held | recall@5 | recall@10 | complete@5 | MRR@10 |
|---|---|---|---|---|---|
| pdftotext 24.02, terms as written | 0.733 | 0.111 | 0.111 | 0.111 | 0.042 |
| pdftotext 26.01, terms as written | 0.533 | 0.111 | 0.111 | 0.111 | 0.042 |
| pdftotext 24.02, full analyzer | 0.733 | 0.417 | 0.583 | 0.333 | 0.485 |
| pdftotext 26.01, full analyzer | 0.533 | 0.333 | 0.426 | 0.222 | 0.387 |
| pipeline, terms as written | 1.000 | 0.611 | 0.630 | 0.611 | 0.438 |
| + normalization | 1.000 | 0.741 | 0.815 | 0.667 | 0.519 |
| + stopwords | 1.000 | 0.741 | 0.815 | 0.667 | 0.531 |
| **+ light stemming** | **1.000** | **0.778** | **0.870** | **0.722** | **0.687** |

A naive pipeline puts 15% of the development evidence among its first five
results, and Daleel puts 84% there. On the held-out questions the figures are
11% and 78%.

## 1. What extraction buys

**pdftotext's text lacks a quarter to a third of the evidence.** Of the 94
development quotes, its text holds 71 with poppler 24.02 and 64 with 26.01.
In the 24.02 run, the missing quotes are on these pages:

| Document | Quotes held by pdftotext 24.02 | Why the rest are missing |
|---|---|---|
| Library services | 2 of 17 | PowerPoint export: letters substituted and scattered |
| Guidance manual | 0 of 4 | Word export: letters substituted |
| Orientation 1446 | 0 of 1 | Word export: letters substituted |
| Organizational regulations | 8 of 10 | 2 on pages drawn without text |
| Student guide 2025 | 33 of 34 | 1, a formula whose parts it puts in another order |
| Student portal guide | 9 of 9 | |
| Academic calendar | 10 of 10 | |
| Student charter | 5 of 5 | |
| Code of student conduct | 4 of 4 | |

These are the failure modes the inventory found before any retrieval was
built ([README](../../README.md#why-this-is-not-a-chat-with-your-pdf-project)).
The quality gate sends those pages to OCR, and the pipeline's chunks hold
all 94 quotes.

**With the same analyzer, extraction is worth 18 to 24 points of
recall@5.** pdftotext with the full analyzer reaches 0.662 with poppler 24.02
and 0.597 with 26.01, against 0.841 for the pipeline.

**The naive baseline depends on the poppler version.** From the same PDFs,
26.01 holds 7 fewer development quotes than 24.02 and 6 fewer of the 30
held-out ones. The calibration had already seen poppler's word order change
between releases ([gate calibration](gate_calibration.md), section 1).

**A correction:** an earlier version of this table matched quotes
character for character, and found that pdftotext held only 50 of the 94
development quotes. pdftotext writes the brackets of right-to-left text the
other way round, as `)(20%` for `(20%)`, and moves the spaces around digits
and slashes, as in `15وحدة`. 21 quotes whose every word it reads correctly
counted as missing, and the gap the extraction closes looked twice as wide.
Quotes are now matched by their words. The pipeline's own numbers did not
change.

## 2. What each step of the analyzer buys

- **Without normalization, pdftotext's Arabic is mostly glyphs a student
  cannot type.** In six of the nine documents, most of its Arabic letters
  are presentation forms (README, table T1), which share no code points with
  typed Arabic. With
  poppler 24.02, it puts some evidence in the first five for 9 questions.
  Seven of them are found through numbers and English words: five calendar
  questions match on 481, 2026 and 2027, and two English questions match the
  calendar's and the portal's English labels. The other two find the
  library's borrowing table, which pdftotext reads as base letters, through
  plain words such as كتاب and مدة.
- **Normalization buys 18 points of recall@5 on the pipeline's own text**,
  from 0.569 to 0.752, although pypdfium2 has already removed the
  presentation forms. It folds what is left: alef with and without hamza,
  taa marbuta and haa, alef maqsura and yaa, diacritics and tatweel, and
  digits in either script.
- **Stopwords buy nothing measurable.** They change recall@5 by 0.006 (one
  quote) and MRR@10 by -0.010. BM25's inverse document frequency already
  gives function words little weight. They stay in the analyzer for now and
  will be measured again with fusion.
- **Light stemming buys 8 points,** from 0.758 to 0.841. It makes الكلية
  and بالكليات one term, and المقررات and مقرر another. Lucene's stemmer
  makes errors, such as taking the و off وقتها, and on this corpus they cost
  less than what it joins.

## 3. Where BM25 fails

On the development questions, with the full analyzer:

| Type | Questions | recall@5 | complete@5 |
|---|---|---|---|
| single_clause | 19 | 0.947 | 0.947 |
| numeric | 11 | 0.909 | 0.909 |
| table_lookup | 6 | 0.833 | 0.833 |
| cross_document | 5 | 0.800 | 0.600 |
| multi_clause | 7 | 0.726 | 0.286 |
| english_query | 3 | 0.667 | 0.667 |
| precedence | 3 | 0.444 | 0.333 |

13 of the 54 questions do not have all their evidence in the first five, and
they fail in four ways:

- **Several parts, one found** (g042 to g045, g059, g060). When a question
  asks for conditions from several clauses, its best-matching clause ranks
  near the top and the rest rank lower. In g045 the clause on full-time
  training ranks first and the one on reports ranks 17th. This is why
  multi_clause has recall@5 0.726 but complete@5 only 0.286.
- **The student's words are not the page's** (g019, g038, g046). The
  question says ساعة معتمدة where the regulations say وحدة دراسية, and
  أتدرب قبل أخلص where the guide says التفرغ الكامل للبرنامج. BM25 only matches
  the words the two share.
- **The question names documents, not the rule** (g065, g066). "دليل التهيئة
  1446 ودليل التوجيه يذكران..." matches the guides' introductions, which repeat
  their own titles, ahead of the attendance clauses.
- **English** (g078). None of the question's terms appear in the Arabic text.

One miss belongs to the gold set, not to BM25. **g056** asks how many books a
student may borrow and for how long. BM25 ranks first the library guide's
own FAQ on page 9, "عدد (5) كتب لمدة أربعة عشريوماً", which answers the
question. The gold set cites only the borrowing table on page 8, which ranks
eighth. Page 9 is a candidate for a gold_v2, together with the third part of
g043, which page 9 answers too.

How a question was written makes little difference. On the development
questions, the 20 asked by classmates have recall@5 0.879, and the 31 written
to cover the corpus have 0.833. The authored questions are not easier for
BM25.

## T4

Over the 54 development questions the corpus answers. Every row holds all
the evidence (*held* 1.000), so that column is left out. The times are per
question, from the question's text to the ranked chunks, on machine A's GPU.

| | recall@5 | recall@10 | complete@5 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25, full analyzer (T3) | 0.841 | 0.897 | 0.759 | 0.681 | 0.6 | 1.0 |
| bge-m3 | 0.806 | 0.829 | 0.704 | 0.661 | 28.6 | 38.9 |
| multilingual-e5-large-instruct | 0.795 | 0.894 | 0.722 | 0.695 | 22.7 | 38.4 |
| RRF: BM25 + bge-m3 | 0.873 | 0.966 | 0.796 | 0.761 | 29.2 | 39.4 |
| RRF: BM25 + e5 | 0.878 | 0.941 | 0.796 | 0.784 | 23.3 | 39.0 |
| RRF: BM25 + both | 0.863 | 0.909 | 0.759 | 0.775 | 54.0 | 72.5 |
| **RRF: BM25 + bge-m3, reranked** | **0.931** | **0.960** | **0.852** | **0.811** | **237.5** | **298.0** |
| RRF: BM25 + e5, reranked | 0.906 | 0.931 | 0.815 | 0.788 | 210.0 | 271.5 |
| RRF: BM25 + both, reranked | 0.935 | 0.941 | 0.870 | 0.808 | 241.5 | 309.6 |

The row in bold is the configuration chosen in section 5. The same rows over
the 18 held-out questions, run once after that choice:

| | recall@5 | recall@10 | complete@5 | MRR@10 |
|---|---|---|---|---|
| BM25, full analyzer (T3) | 0.778 | 0.870 | 0.722 | 0.687 |
| bge-m3 | 0.796 | 0.884 | 0.722 | 0.634 |
| multilingual-e5-large-instruct | 0.704 | 0.903 | 0.667 | 0.679 |
| RRF: BM25 + bge-m3 | 0.829 | 0.903 | 0.778 | 0.760 |
| RRF: BM25 + e5 | 0.833 | 0.903 | 0.833 | 0.729 |
| RRF: BM25 + both | 0.829 | 0.958 | 0.778 | 0.747 |
| **RRF: BM25 + bge-m3, reranked** | **0.907** | **0.940** | **0.833** | **0.753** |
| RRF: BM25 + e5, reranked | 0.907 | 0.944 | 0.833 | 0.738 |
| RRF: BM25 + both, reranked | 0.907 | 0.926 | 0.833 | 0.731 |

Fused and reranked, the first five chunks hold 93% of the development
evidence, against 84% for BM25, and every piece of it for 46 of the 54
questions, against 41. On the held-out questions the figures are 91% against
78%, and 15 of 18 questions against 13.

## 4. What each stage buys

**Alone, neither encoder beats BM25 on these questions, but each finds
evidence BM25 misses.**
bge-m3 puts 0.806 of the development evidence in the first five, and e5
0.795, against BM25's 0.841. But 7 questions have all their evidence in
BM25's first five and not in bge-m3's (g002, g008, g016, g041, g062, g063,
g068), and 4 the other way round (g019, g044, g056, g078). Those 4 are
misses of BM25 from section 3, where the student's words are not the page's
or the question is in English.

**Fusion widens the pool more than it sharpens the top.** BM25 fused with
bge-m3 lifts recall@10 from 0.897 to 0.966, and recall@5 only from 0.841 to
0.873, with 7 questions gaining recall@5 and 5 losing it. What fusion buys
is mostly in ranks 6 to 20, where a reranker can reach it.

**The reranker turns the pool into the first five.** Reranking the fused
first 20 lifts recall@5 from 0.873 to 0.931, complete@5 from 0.796 to 0.852,
and the share of evidence ranked first from 0.506 to 0.603. 5 development
questions gain recall@5 and 1 loses it. On the held-out questions it lifts
recall@5 from 0.829 to 0.907: two questions gain all their evidence in the
first five, and two lose a part of theirs. It is also the slow stage:
reranking 20 chunks takes 206 of the 238 ms the chosen configuration needs
per question at the median, and 255 of 298 ms at the 95th percentile. The
design asked whether the reranker pays for its latency. A fifth of a second
per question, before an answer is even generated, buys 6 points of recall@5
on the development questions and 8 on the held-out ones. It stays.

**Reranking BM25 alone gets most of the way.** Without an encoder, the
reranked BM25 ranking reaches recall@5 0.912 and complete@5 0.833 on the
development questions, against 0.931 and 0.852 with fusion, and 0.852 and
0.778 on the held-out ones, against 0.907 and 0.833. The encoder's part is
the questions whose evidence BM25 does not rank in its first 20 at all:
g038 and g078 among the development questions, g077 among the held-out
ones. It costs g008, below.

**English questions need the encoder.** Of the 4 in English (3 for
development, 1 held out), BM25 finds evidence for 2, through numbers and the
English labels of the calendar and the portal, and none for g077 and g078.
Fused and reranked, all 4 have their evidence in the first five. The design
also names a second way, translating the question into Arabic before
retrieving. It needs the generator, and with 4 English questions neither
way can be shown better than the other.

## 5. Choosing the configuration

The configuration was fixed on the development questions on 7 October,
before the held-out questions were run: BM25 + bge-m3, fused over the first
100 of each, with the first 20 reranked.

- **One encoder.** Fusing both encoders with BM25 and reranking gains one
  development question (complete@5 0.870 against 0.852) and costs a second
  model and its encoding, about 22 ms per question. One question is not a
  difference.
- **bge-m3 rather than e5.** Reranked, bge-m3's fusion is ahead on recall@5
  for 2 questions and behind for none. Before reranking, e5's fusion has the
  better MRR@10, 0.784 against 0.761. The evidence is thin. The timing,
  measured after the choice, favours e5: its reranked row took 210 ms at the
  median against 238. The encoders take about the same time, 16 and 22 ms to
  encode a question, and most of the difference is in reranking, whose time
  follows the length of the chunks it reads.
- **Reranking 20.** The design set 20 before anything was measured.
  Reranking 10 did as well on the development questions, one question
  better (recall@5 0.935, complete@5 0.870), in 136 ms at the median instead
  of 238. But a setting changed on one question is a setting tuned to the
  noise, so the design stands.

On the held-out questions, the choices between near ties made no
difference except one. Both encoders, or e5 instead of bge-m3, give the same
recall@5 and complete@5 there (0.907 and 0.833). Reranking 10 would have
given 0.852 and 0.778, a question fewer.

## 6. Where the hybrid still fails

8 of the 54 development questions still lack some evidence in the first
five, against 13 for BM25. Seven of BM25's misses are found (g019, g038,
g043, g044, g056, g060, g078), and two questions BM25 answered in full are
lost (g008, g041).

- **Parts outside the first 20** (g042, g045, g046, g059). Some part of the
  evidence is not among the fused first 20 at all, so the reranker never
  reads it. These questions ask for several clauses at once, and one ranking
  per question finds the clause that matches the question as a whole.
  Splitting such a question into its parts belongs to the query rewriting
  step of the answering pipeline.
- **Documents named instead of the rule** (g065, g066). Their evidence moves
  up, in g065 from ranks 14, 16, 6 and none to 6, 5, 1 and 4, but one part
  stays just outside the first five.
- **A list without its heading** (g008). The clause listing accepted excuses
  begins "1- في حال التنويم" and never says عذر. The word is in its heading,
  "شروط قبول اعذار الغياب", which is the chunk's metadata, and neither the
  encoders nor the reranker read it. BM25 ranks the clause first on the
  words الإصابة بمرض, which only two chunks contain. The reranker ranks it
  sixth, behind chunks about excuses in general.
- **The rule's other source** (g041). The gold set cites the student guide's
  summary of re-marking a final exam, on pages 59 and 60. The reranker ranks
  two chunks of the regulations' own article on it, article 65 on page 25 of
  the organizational regulations, third and fourth, ahead of the cited
  pages. The article answers the question too, and differs from the guide in
  one detail: the regulations finish the review within five working days of
  the request, the guide within a week. It is a candidate for gold_v2, with
  g056's page 9.

## 7. Limits

- **Small samples.** One development question is 1.9 points of recall@5, and
  one held-out question is 5.6. Differences smaller than a question or two,
  such as the stopwords row and the choices between near ties in section 5,
  are noise.
- **Holding the words is not answering.** A chunk counts if it contains the
  quote's words. Whether a model can answer from it is measured on answers.
- **Evidence is what the gold set cites.** Another page with the same answer
  counts as a miss, as g056 shows. Such cases are gathered for a gold_v2,
  reported beside v1, and v1 is not changed.
- **One extraction.** Both machines chunk the records extracted once on
  machine A. OCR is cached and was not repeated on machine B.
- **One GPU.** The vectors and scores were computed once, in half precision.
  Another GPU or precision would give slightly different numbers, so T4 is
  reproduced from the stored files rather than from the models. In half
  precision, the reranker gives two different chunks of a question's first
  20 the same score 8 times over the 80 questions, and tied chunks keep
  their fused order.
- **Latency on one machine.** Times are for one question at a time, with the
  models loaded and warm, the stages run one after another, and WSL 2
  between the code and the GPU. Loading the models is not counted. The same
  row varies between runs: bge-m3 alone took 28.6 ms at the median in one
  run and 20.1 in another, most of the difference in searching 768 vectors,
  which took 11.9 ms in the first and 2.3 in the second for the same work.
  Differences of 10 ms or so between rows are within that.

## Reproduce

```bash
daleel chunk                                             # data/processed/chunks.jsonl
python3 scripts/ablate_normalization.py --split dev      # T3, development questions
python3 scripts/ablate_normalization.py --split final    # T3, held-out questions
python3 scripts/eval_retrieval.py --split dev --misses   # by type and language, and the misses
uv pip install -e ".[dense]"                             # T4: the models' libraries
python3 scripts/encode_dense.py                          # vectors, in data/interim/dense/
python3 scripts/score_rerank_pool.py                     # reranker scores, in data/interim/rerank/
python3 scripts/eval_hybrid.py --split dev               # T4, development questions
python3 scripts/eval_hybrid.py --split final             # T4, held-out questions
python3 scripts/time_retrieval.py --dense bge-m3 --dense e5-large-instruct --rerank
```

The pdftotext rows need poppler-utils and the PDFs in `data/raw/`. The last
line of the script's output names the poppler version. The T4 scripts after
the first two need neither the models nor a GPU, only the stored files, and
print the models' revisions and the device that made them.
