# Daleel · دليل

[![CI](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml/badge.svg)](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml)

Answers questions about Royal Commission for Jubail and Yanbu college
regulations in Arabic, with the exact clause cited.

On the 54 development questions the corpus answers, a naive pipeline
(poppler's `pdftotext`, BM25 on words as written) puts 15% of the evidence
among its first five results. Daleel's extraction, chunking and Arabic
analyzer put 84% there ([table T3](eval/results/retrieval.md)), and fusing
BM25 with a multilingual encoder and reranking the result puts 93% there
([table T4](eval/results/retrieval.md#t4)).

> **Status: in development.** This README grows with the repository.
> No result is published here until it has been measured. See
> [Results](#results).

## The problem

Students need answers from 257 pages of Arabic regulation documents:
*Can I transfer majors? What GPA do I need? When is the drop/add deadline?*
Today the options are reading all of it, asking a classmate who may be wrong,
or waiting for the registrar.

An unsourced answer about your own academic standing is unusable: you cannot
act on it and the registrar cannot confirm it. So **every answer this system
gives carries a document, page and clause reference.** Citation is a hard
functional requirement, not a feature.

## Why this is not a "chat with your PDF" project

Measuring the corpus before building anything turned up extraction damage that
no naive pipeline would notice. `daleel inventory data/raw/` reports it, and
`--backend pdfplumber` gives the first presentation-form column:

| Document | Producer | Pages | Chars per page | Presentation forms, pdfplumber | Presentation forms, pypdfium2 |
|---|---|---|---|---|---|
| Student Guide 2025 | PDFium | 86 | 1,297 | 77% | 0% |
| Academic weeks 1448 | Adobe PDF library 18.00 | 3 | 1,637 | 71% | 0% |
| Guidance manual | Microsoft® Word 2019 | 19 | 566 | 0% | 0% |
| Library services | Microsoft® PowerPoint® 2019 | 16 | 320 | 0% | 0% |
| Orientation 1446 | Microsoft® Word 2019 | 13 | 202 | 0% | 0% |
| Organizational regulations | none | 58 | 1,445 | 76% | 0% |
| Code of student conduct | Adobe PDF library 15.00 | 12 | 1,665 | 75% | 0% |
| Student charter | Adobe PDF library 15.00 | 8 | 900 | 76% | 0% |
| Student portal guide | GPL Ghostscript 10.00.0 | 42 | 387 | 77% | 0% |

Five failure modes sit behind those numbers:

1. **Presentation-form substitution.** In six of the nine documents, most
   Arabic extracts as Unicode Arabic Presentation Forms, the contextual glyph
   variants a renderer picks per letter position, rather than as base letters,
   at least through pdfplumber and pdftotext (see below). A student typing
   `نظام` cannot match an index holding `ﻧﻈﺎم`: different codepoints, so zero
   term overlap.
2. **Silently wrong text.** The Word and PowerPoint exports report 0%
   presentation forms and look clean by every structural measure, but the text
   is wrong: `رؤيتنا` ("our vision") extracts as `وؤيتنا`, with letters
   substituted and dropped. These are valid Arabic codepoints that are not
   real words, so nothing structural can flag them. Verified by rasterising
   pages and reading them against the text layer.
3. **Low text density.** The PowerPoint and orientation documents yield about
   200 to 320 characters per page, meaning most of their content is graphics
   and needs OCR regardless of whether the extracted text is sound.
4. **Text drawn as shapes.** Design tools can convert text to vector outlines,
   leaving no text layer at all. 25 of the 257 pages have no Arabic in their
   text layer, from covers to the regulations' page defining every grade code,
   and only OCR can read them.
5. **Glyph codes read as characters.** One page embeds a form whose fonts carry
   no usable character map, so its text layer holds Ethiopic and Canadian
   Syllabics letters instead of Arabic.

### The failure signature depends on the extraction backend

The first five documents, read with pdfplumber and with poppler's `pdftotext`:

| Document | pdfplumber chars | pdftotext chars | Presentation forms | pdftotext bidi controls /1k |
|---|---|---|---|---|
| Student Guide 2025 | 109,280 | 120,157 | 77% | 93.5 |
| Academic weeks 1448 | 4,811 | 5,776 | 71% | 156.5 |
| Guidance manual | 10,629 | 12,976 | 0% | 126.1 |
| Library services | 4,924 | 5,624 | 0% | 121.3 |
| Orientation 1446 | 2,595 | 3,089 | 0% | 139.9 |

The presentation-form ratio agrees between these two, but a third extractor,
pypdfium2, returns base letters for the same glyphs, so the ratio describes how
an extractor maps glyphs rather than the PDF alone. Word order differs too:
pdfplumber writes Arabic in visual order, spelling every word backwards, which
is why the pipeline extracts text with pypdfium2
([calibration](eval/results/gate_calibration.md)).

`pdftotext` also writes a form feed after every page. Once those and its bidi
marks are subtracted, the two backends agree on text length within 2% on four
documents and 6.5% on the guidance manual. Quality gate thresholds are
therefore only meaningful per backend, and must be recorded with one.

Scaled pages break pdfplumber's word grouping. The regulations, the conduct
code and the charter define their pages at one tenth of A4, and pdfplumber
groups characters into lines and words with fixed tolerances sized for standard
pages. On these it merges lines and splits words: page 4 of the regulations
gives 1,690 words through pdfplumber and 359 through pypdfium2, from the same
1,846 letters. With its tolerances scaled to the page, pdfplumber gives 360.
`scripts/inspect_pages.py` shows this for any page.

### Provenance

The 1448 academic calendar first added to the corpus turned out, on reading its
footer, to be an unofficial student redesign that states it does not represent
the Royal Commission. It was replaced with the official calendar. The
unofficial file was also the only source of real control-character corruption
seen so far, and of a large unexplained disagreement in character counts
between the two backends. Both left the corpus with it.

A second derivative turned up later: an AI-generated `.ics` version of the
full-year calendar, identified by its metadata. All 37 of its dates match the
official calendar, but it rewrote one summer deadline from the end of drop and
add to the end of add, which would mislead a student. It is used only to
cross-check the calendar ground truth's dates, and is never cited.

The organizational regulations first arrived as a 70-page copy shared on
WhatsApp, with metadata that looked entirely official. Comparing it against the
college's site showed two documents bound together: the site publishes the
58-page regulations and the 12-page code of student conduct separately, and
both are in the corpus as published. A copy of the conduct code saved through a
browser's print dialog was likewise replaced with the original download. The
student guide, which the site does not offer for download, comes from the
Student Affairs department's Telegram channel, where the post has since been
removed.

Every document's SHA-256 is recorded in [`data/SHA256SUMS`](data/SHA256SUMS),
and `sha256sum -c data/SHA256SUMS` checks a local copy against it.

## Extraction routing

`daleel route data/raw/` predicts, for each document, which extraction path to
try first and which failure to expect, from the software recorded in its
metadata:

| Document | Producer | Path | Expected failure |
|---|---|---|---|
| Student Guide 2025 | PDFium | text layer | presentation forms |
| Academic weeks 1448 | Adobe PDF library 18.00 | text layer | presentation forms |
| Guidance manual | Microsoft® Word 2019 | OCR | lossy substitution |
| Library services | Microsoft® PowerPoint® 2019 | OCR | lossy substitution |
| Orientation 1446 | Microsoft® Word 2019 | OCR | lossy substitution |
| Organizational regulations | none | text layer | unknown (default) |
| Code of student conduct | Adobe PDF library 15.00 | text layer | presentation forms |
| Student charter | Adobe PDF library 15.00 | text layer | presentation forms |
| Student portal guide | GPL Ghostscript 10.00.0 | text layer | unknown (default) |

A route is a prior, not a verdict. The quality gate still checks every
document, so a wrong prediction costs time and never correctness.

The rules come from the first five documents, and each rule in the code names
the documents it rests on. All three Microsoft documents have since been checked
against hand-transcribed ground truth, and all three text layers are broken:
their words match the page at 5%, 50% and 33% precision. Every route held,
including the orientation guide's, which had been a prediction only. Software
with no evidence behind it, including other Microsoft and Adobe products,
falls to a default: try the text layer and let the gate decide.

The four documents added later tested the rules out of sample. The conduct code
and the charter use Adobe PDF library 15.00, a version no rule was built from,
and routed correctly. The regulations carry no metadata at all, and an online
compressor rewrote the portal guide's producer and creator, so both took the
default, and the gate then trusted their text layers.

## Quality gate

`daleel gate data/raw/` judges every page: can its text layer be trusted, or
must the page go to OCR? It extracts text with pypdfium2, normalizes it, and
checks its words against a web-scale Arabic word list, with thresholds
calibrated against the hand-transcribed ground truth.

| Document | Route | Pages | Trusted | Untrusted | No Arabic text | Agrees |
|---|---|---|---|---|---|---|
| Academic weeks 1448 | text layer | 3 | 3 | 0 | 0 | yes |
| Guidance manual | OCR | 19 | 1 | 18 | 0 | yes |
| Library services | OCR | 16 | 1 | 15 | 0 | yes |
| Orientation 1446 | OCR | 13 | 1 | 12 | 0 | yes |
| Student Guide 2025 | text layer | 86 | 82 | 2 | 2 | yes |
| Organizational regulations | text layer | 58 | 43 | 1 | 14 | yes |
| Code of student conduct | text layer | 12 | 9 | 0 | 3 | yes |
| Student charter | text layer | 8 | 3 | 0 | 5 | yes |
| Student portal guide | text layer | 42 | 37 | 4 | 1 | yes |

The router predicts each route from metadata alone; the gate reaches the same
conclusion for all nine documents by reading every word. Pages with no Arabic
in their text layer go to OCR whatever their document's route.

The gate still trusts two visibly broken pages, because their fragments are
real entries in a web word list, and it rejected two sound pages of the portal
guide over rare real words. Those limitations, the evidence behind every
threshold, and how to reproduce each number are in
[`eval/results/gate_calibration.md`](eval/results/gate_calibration.md).

## Extraction

`daleel extract data/raw/` turns every page into a record: its text, the
method that produced it, and the gate's evidence for the choice. A page keeps
its text layer when the gate trusts both the page and its document as a whole.
Every other page is read by OCR with dots.mocr, the most accurate engine in
[`eval/results/ocr.md`](eval/results/ocr.md), so the two broken pages the gate
still trusts are read by OCR too, since their documents go to OCR. An OCR record
also keeps the page's layout: each block's category, text and box, in the
page's own points, in reading order. dots.mocr reads parts of a page that sit
side by side left to right, so those are put right to left. It also writes two
Arabic spellings with letters of other scripts, the Thai ส for سب in أسبوع and
the Hebrew ת for ت, and those are repaired inside Arabic words. A page whose
OCR text holds less than half of the Arabic words in its text layer has lost
text, most often inside what its layout calls a picture, so it is read again
for its text alone, and what that reading adds goes in after the picture. How
the library deck's records were checked against its slides, and what that led
to, is in [`eval/results/layout.md`](eval/results/layout.md).

OCR needs dots.mocr served by vLLM, started as in
[`eval/results/ocr.md`](eval/results/ocr.md), and the same limits given to the
command:

```bash
daleel extract data/raw/ --quantization fp8 --max-pixels 5400000
```

Records go to `data/interim/extracted/<document>.jsonl`, a line per page.
Every OCR reading is kept in `data/interim/ocr_cache/`, keyed on the engine and
the page image, so a second run reads no page again, and an interrupted run
resumes where it stopped.

The whole corpus has been extracted. Of its 257 pages, 177 kept their text
layer, and dots.mocr read the other 80 in 48 minutes on the machine described
in [`eval/results/ocr.md`](eval/results/ocr.md): 22 seconds a page at the
median, and 139 seconds at most, for a page of the regulations drawn without a
text layer. Two library slides were read again for their text.

## Calendar

The academic calendar answers what students ask most often: when drop and add
closes, when exams start. The gate trusts its text layer, and PDFium reads
every event's lines whole, every date intact. What a page of text loses is the
grid: the event cards come in the order they were drawn, so a date can sit
beside another event's title. OCR is no way around it, since dots.mocr
invented a second date in every date box.

`daleel calendar` rebuilds each card from where its lines sit on the page.
Every card prints exactly one Gregorian date, so each date anchors a card, and
the lines in its column, from the title above it down to the Hijri date below
it, are that card's. Cards are read top to bottom and each row right to left,
as the annotation guidelines read a page. Two faults of the text layer are
mended: its English font writes the fi ligature as the control character
U+001F, and five times a word lost the space before it, as in `الدراسيالأول`.
A word the gate's word list does not hold, which splits exactly one way into
two words it does, is split.

```bash
daleel calendar data/raw/academic_weeks_1448.pdf
python3 scripts/check_calendar_csv.py data/interim/calendar/academic_weeks_1448_p01.csv <ics> 14
```

Each page's rows go to `data/interim/calendar/<document>_p<NN>.csv`, in the
ground truth's columns: the title, the days, each in Arabic and in English,
then the Gregorian and the Hijri date as printed.

Page 1 came out identical to its hand-transcribed ground truth, all 84 cells
of its 14 rows. Pages 2 and 3 have no ground truth, so the checks stand in for
it, and every row passes them: its Gregorian dates are an event's dates in the
.ics, its two calendars agree on how long the event lasts, and its English days
are the days its dates fall on. The three pages give 37 rows, and their dates
pair off one to one with the 37 events of the .ics, the AI-generated copy
described under [Provenance](#provenance), trusted for its dates and nothing
else.

The later pages' quirks stay as printed, as ground truth keeps them: page 2's
`Semestet`, page 3's `حذف و إضافة`, months and days of one digit, a space
before a date's suffix, and a Hijri range over two lines, each a full date
with its own suffix. The checks read all of these.

## Chunking

`daleel chunk` cuts the records into chunks along the documents' own
structure, since a chunk cut through the middle of a clause cites nothing. A
heading is a title or section header in an OCR page's layout, an article's
label, a numbered part such as `ثانياً:`, or one of the sections the student
guide's table of contents lists. A unit starts at a numbered clause (`.1`,
`1-`, `1)`), a bullet or a layout block. Running headers, page numbers and
pictures are left out. Units shorter than 40 words merge with their
neighbours under the same heading on the same page, and units longer than 180
are split at a sentence end, with 30 words of overlap. A chunk never spans two
pages, so it always cites a single page.

The regulations' text layer prints each article's label at the end of the
article's first line, because the margin holding the label is read with that
line. Those labels are taken as headings when they end in a colon or follow
the last article's number. A passing reference such as `في المادة الخامسة` is
not taken as a heading. This finds 71 of the regulations' 72 articles and all
37 of the conduct code's. The one it misses, article 1, has no label in the
extracted text.

A table becomes one sentence per row, with each value named by its column, so
a retrieved row can be cited on its own: `الفئة: الطلبة، عدد الكتب: 5 كتب، مدة
الاعارة: 14 يوم`. Tables come from three places. An OCR page's layout gives
its tables as HTML, and a merged cell is repeated in every row it covers. The
academic calendar's 37 cards become a chunk each, rebuilt as
[Calendar](#calendar) describes and named with their semester, so `daleel
chunk` reads the calendar's PDF as well as the records. And one table's text
layer lists its amounts apart from their rows: the fee table on page 28 of
the student guide. It was typed in by hand from the rendered page, and is in
[`data/manual/tables.json`](data/manual/tables.json) with the lines of the
text layer it replaces. Its chunk is marked `manually_verified`.

```bash
daleel chunk    # data/interim/extracted/ to data/processed/chunks.jsonl
```

The 257 pages give 768 chunks. Every chunk carries what a citation needs: the
document's title, scope and date, the chunk's page, heading and clause number,
whether it is a clause, prose or a table, and how its page was extracted, so a
wrong answer can be traced back to its page.

## Gold set

[`eval/gold_v1.jsonl`](eval/gold_v1.jsonl) holds the 80 questions that
retrieval and answers are measured on: 28 asked by classmates, one by the
author, 47 written to cover the corpus, and 4 English translations. They
come in eight types, from single clauses and numbers to questions the corpus
cannot answer. Each question names the pages that answer it and quotes the
words on them that hold the answer. Every answer was drafted from the
extracted text, then checked by hand against its PDF page. The review changed
six of them, and the set was frozen on 6 October 2026. A mistake found later
will go into a gold_v2, reported beside it.

Twenty questions are held out for the final comparison and not looked at
during development. They are chosen in whole topic groups, so no paraphrase
of a held-out question is among the other 60, and each type is held out in
its share of the set (`choose_final` in `daleel.eval.gold`, seed 1448). Of the
72 questions the corpus answers, 54 are for development and 18 are held out.

## Retrieval

The gold set names pages and quotes, not chunks, so the evidence a question
needs is found anew for every way of chunking: a chunk holds a quote when the
quote's words, normalized, appear in its text one after another, whatever the
punctuation and spacing between them, and a quote given as a group of parts
holds only where all of them are in one chunk. A question needs every quote
on every page that answers it. `scripts/eval_retrieval.py` searches the
chunks for each of the 72 questions the corpus answers and reports recall
and completeness at 1, 5, 10 and 20, and MRR@10, overall, by question type
and by language.

The first retriever is BM25 with Lucene's parameters over an Arabic
analyzer: normalization for comparison, stopwords (with the colloquial
question words of Saudi Arabic), and Lucene's light stemmer. Each step can
be turned off, to measure what it buys: `scripts/ablate_normalization.py`
adds them one at a time, and sets the pipeline's chunks beside a naive
pipeline's, poppler's `pdftotext` read word for word, chunked the same way
(table T3). On the 54 development questions:

| BM25 | recall@5 | complete@5 |
|---|---|---|
| pdftotext, words as written | 0.154 | 0.148 |
| pdftotext, full analyzer | 0.597 to 0.662 | 0.500 to 0.593 |
| pipeline, words as written | 0.569 | 0.519 |
| + normalization | 0.752 | 0.685 |
| + stopwords | 0.758 | 0.685 |
| + light stemming | 0.841 | 0.759 |

The pdftotext rows depend on the poppler version: 26.01 and 24.02 give the
two ends of each range. Its text lacks a quarter to a third of the
evidence, almost all of it on pages the quality gate sends to OCR.
Normalization is the largest single step, and stemming the next; stopwords
change nothing measurable. The 13 questions without all their evidence in
the first five ask for several clauses at once, word things as students do
rather than as the regulations do, name documents instead of the rule, or
ask in English.
[`eval/results/retrieval.md`](eval/results/retrieval.md) has the held-out
questions, every miss, and the limits.

Dense retrieval is measured the same way (table T4). Two multilingual
encoders, bge-m3 and multilingual-e5-large-instruct, map questions and chunks
to vectors; BM25 is fused with each and with both by reciprocal rank fusion;
and a cross-encoder, bge-reranker-v2-m3, reorders the first 20 of each
fusion. The models want a GPU that the evaluation need not have, so
`scripts/encode_dense.py` and `scripts/score_rerank_pool.py` compute the
vectors and scores once and keep them in `data/interim/`, each under a hash
of the text it came from, with the model, its revision, the library versions
and the device. `scripts/eval_hybrid.py` measures every row from those alone.
On the 54 development questions, with times per question on an RTX 3060 Ti:

| | recall@5 | complete@5 | p50 ms | p95 ms |
|---|---|---|---|---|
| BM25 | 0.841 | 0.759 | 0.6 | 1.0 |
| bge-m3 | 0.806 | 0.704 | 28.6 | 38.9 |
| BM25 + bge-m3, fused | 0.873 | 0.796 | 29.2 | 39.4 |
| **BM25 + bge-m3, fused and reranked** | **0.931** | **0.852** | **237.5** | **298.0** |

On these questions neither encoder alone beats BM25, but an encoder and
BM25 miss different questions, so fusing them gives the reranker more of the
evidence to reorder. The reranker makes most of the gain and most of the wait. The
last row was chosen on the development questions; on the 18 held out, run
once afterwards, it puts 91% of the evidence in the first five, against
BM25's 78%. Of the 4 English questions, BM25 finds evidence for 2, and the
last row for all 4.

## Usage

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 scripts/fetch_lexicon.py                  # the gate's word list: 69 MB, not committed
daleel inventory data/raw/                        # text-layer measurements, via pypdfium2
daleel inventory data/raw/ --backend pdfplumber   # the same, via pdfplumber, for comparison
daleel route data/raw/                            # predicted extraction path per document
daleel gate data/raw/                             # verdicts per document
daleel gate data/raw/ --pages                     # verdicts per page, with reasons
daleel extract data/raw/                          # every page's text as a record (OCR: see above)
daleel calendar data/raw/academic_weeks_1448.pdf  # the calendar's events, a row for each
daleel chunk                                      # the records cut into chunks, with their metadata
sha256sum -c data/SHA256SUMS                      # check your copies of the documents
python3 scripts/inspect_pages.py <pdf> <pages>    # what chosen pages are made of
python3 scripts/check_calendar_csv.py <csv> <ics> # check a page of calendar rows
python3 scripts/check_gold.py --records           # check the gold set, its quotes against the pages
python3 scripts/eval_retrieval.py --misses        # BM25 measured against the gold set's evidence
python3 scripts/ablate_normalization.py           # table T3: what extraction and each analyzer step buy
uv pip install -e ".[dense]"                      # the encoders and the reranker (see the dependency note)
python3 scripts/encode_dense.py                   # chunks and questions as vectors, kept for T4
python3 scripts/score_rerank_pool.py              # the reranker's scores for every candidate, kept for T4
python3 scripts/eval_hybrid.py                    # table T4: the encoders, fusion and reranking
python3 scripts/time_retrieval.py --dense bge-m3  # time each stage per question; --rerank adds the reranker
python3 scripts/ask_llm.py "سؤال"                 # one question to the answering model, stored for reruns
python3 scripts/ocr_eval.py tesseract             # score an OCR engine against the ground truth
python3 scripts/ocr_eval.py saved <run folder>    # score a saved OCR run again
pytest                                            # the test suite
```

Every `daleel` command also accepts `--json`. `python3 scripts/calibrate_gate.py`
reproduces every number in the gate's calibration, given the hand-transcribed
ground truth, which is not redistributed.

## Dependency note

Text extraction uses **pypdfium2** (BSD-3-Clause or Apache-2.0), chosen by
measurement: pdfplumber writes Arabic in visual order, spelling every word
backwards (see [the calibration](eval/results/gate_calibration.md)). It is
held below 5.13, whose PDFium puts the words of an Arabic line last word first.
Each word is still spelled right, so nothing that counts words notices, and
every command reads a generated Arabic line first and stops on a build that
reorders it.
**pdfplumber** (MIT) remains as a comparison backend for the inventory. The
calendar's grid is rebuilt from PDFium's character boxes instead, which come
with its text in logical order.
PyMuPDF was ruled out because it is AGPL-licensed, a blanket disqualifier at
many organisations that would force this repository's own licence to match.

The `dense` extra pins **sentence-transformers** (Apache-2.0) and
**transformers** (Apache-2.0), so that vectors and scores from different runs
can be compared. It brings PyTorch with it: on a GPU, install the PyTorch
build for your CUDA version first, as <https://pytorch.org> shows, or pip may
bring one that does not use the GPU. The encoders are MIT-licensed and the
reranker Apache-2.0.

## Results

The extraction inventory above (table T1), the quality gate's calibration in
[`eval/results/gate_calibration.md`](eval/results/gate_calibration.md), and the
OCR engines compared in [`eval/results/ocr.md`](eval/results/ocr.md) (table T2):
dots.mocr, a vision language model, reads Arabic about three times as
accurately as Tesseract or PaddleOCR, and invented a second date in every date
box of the academic calendar. The calendar's events are rebuilt from its text
layer instead, and its first page matches its ground truth cell for cell
([Calendar](#calendar)). Retrieval with BM25 is measured in
[`eval/results/retrieval.md`](eval/results/retrieval.md) (table T3): 84% of
the development questions' evidence among the first five results, against
15% for a naive pipeline. Fused with bge-m3 and reranked by
bge-reranker-v2-m3 (table T4), retrieval puts 93% there, and 91% of the
held-out questions' evidence against BM25's 78%, in 238 ms per question
at the median on an RTX 3060 Ti. The arm comparison lands here as that work
completes. Nothing is published here before it is measured.

## Prediction

Stated on 9 October 2026, before any arm that answers questions was run.

The study compares five ways of answering: A, a model without the
documents; B, the whole corpus in the model's context; C, this pipeline,
retrieving clauses for the model to answer from; D, a model fine-tuned on
the corpus, without retrieval; E, the fine-tuned model with retrieval. Each
comparison holds the model fixed. Arms A to C use `gemini-3.5-flash-lite`
through Google's API, and arms A, C, D and E then run on one small
open-weight model, which unlike an API model can be fine-tuned.

1. **Arm A answers almost nothing.** Without the documents the model may
   know the general shape of Saudi college rules, but not these colleges'
   numbers. It will get few numeric questions exactly right, and it will
   state wrong values rather than say it does not know.
2. **Arm B comes close to arm C on accuracy, but not on citations or cost.**
   With the whole corpus in its context it can find most answers, within
   about 10 points of arm C's accuracy. But it has to name the page itself,
   so fewer of its citations will hold the answer, and each question will
   cost it about 50 times arm C's input tokens.
3. **Arm C leads on valid citations and exact numbers,** and declines the
   questions the corpus does not answer.
4. **Retrieval beats fine-tuning on knowledge.** Fine-tuning teaches form
   and behaviour far better than it teaches specific numbers. For a corpus
   of precise thresholds, such as a cumulative average of 3.75 out of 4.00,
   or fees of 250, 750 and 1,100 per credit unit, arm D will answer fluently
   and with wrong values. Arm E may beat arm C on behaviour (register,
   structure, declining what it cannot answer) while matching it on
   knowledge: retrieval for facts, fine-tuning for form.

With 72 answerable questions, differences under about 10 points are not
findings. Each prediction will be marked here as held or not held, with its
numbers, once its arms have run.

## Licence

MIT, see [LICENSE](LICENSE).

## Source documents

The source documents are college publications and are not
redistributed here. See [`data/README.md`](data/README.md) for how to obtain
them.
