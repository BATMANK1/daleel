# Retrieval with BM25 (T3)

This document measures the first retriever, BM25, against the gold set. It
asks how much of the evidence each question needs is ranked among the first
five chunks, and how much of that the extraction and each step of the Arabic
analyzer contribute. It holds table T3. Dense encoders, fusion and reranking
(T4) will be added here once they are measured.

The numbers come from `scripts/ablate_normalization.py` and
`scripts/eval_retrieval.py`, run on two machines with two versions of
poppler. Every design choice rests on the development questions alone. The
held-out questions are reported only for configurations that involve no
tuning.

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
- **Machines:**
  - *A*: Windows with WSL 2, Python 3.12.14, poppler 26.01.0.
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

## 4. Limits

- **Small samples.** One development question is 1.9 points of recall@5, and
  one held-out question is 5.6. Differences smaller than a question or two,
  such as the stopwords row, are noise.
- **Holding the words is not answering.** A chunk counts if it contains the
  quote's words. Whether a model can answer from it is measured on answers.
- **Evidence is what the gold set cites.** Another page with the same answer
  counts as a miss, as g056 shows. Such cases are gathered for a gold_v2,
  reported beside v1, and v1 is not changed.
- **One extraction.** Both machines chunk the records extracted once on
  machine A. OCR is cached and was not repeated on machine B.

## Reproduce

```bash
daleel chunk                                             # data/processed/chunks.jsonl
python3 scripts/ablate_normalization.py --split dev      # T3, development questions
python3 scripts/ablate_normalization.py --split final    # T3, held-out questions
python3 scripts/eval_retrieval.py --split dev --misses   # by type and language, and the misses
```

The pdftotext rows need poppler-utils and the PDFs in `data/raw/`. The last
line of the script's output names the poppler version.
