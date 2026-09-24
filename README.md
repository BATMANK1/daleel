# Daleel · دليل

[![CI](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml/badge.svg)](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml)

Answers questions about Royal Commission for Jubail and Yanbu college
regulations in Arabic, with the exact clause cited.

> **Status: in development.** This README grows with the repository.
> No result is published here until it has been measured see
> [Results](#results).

## The problem

Students need answers from 137 pages of Arabic regulation documents:
*Can I transfer majors? What GPA do I need? When is the drop/add deadline?*
Today the options are reading all of it, asking a classmate who may be wrong,
or waiting for the registrar.

An unsourced answer about your own academic standing is unusable you cannot
act on it and the registrar cannot confirm it. So **every answer this system
gives carries a document, page and clause reference.** Citation is a hard
functional requirement, not a feature.

## Why this is not a "chat with your PDF" project

Measuring the corpus before building anything turned up extraction damage that
no naive pipeline would notice. `daleel inventory data/raw/` reports it:

| Document | Producer | Pages | Chars per page | Presentation forms, pdfplumber | Presentation forms, pypdfium2 |
|---|---|---|---|---|---|
| Student Guide 2025 | PDFium | 86 | 1,297 | 77% | 0% |
| Academic weeks 1448 | Adobe PDF library 18.00 | 3 | 1,637 | 71% | 0% |
| Guidance manual | Microsoft® Word 2019 | 19 | 566 | 0% | 0% |
| Library services | Microsoft® PowerPoint® 2019 | 16 | 320 | 0% | 0% |
| Orientation 1446 | Microsoft® Word 2019 | 13 | 202 | 0% | 0% |

Three distinct failure modes sit behind those numbers:

1. **Presentation-form substitution.** In the two design-tool exports, most
   Arabic is stored as Unicode Arabic Presentation Forms the contextual
   glyph variants a renderer picks per letter position rather than as base
   letters. A student typing `نظام` cannot match an index holding `ﻧﻈﺎم`:
   different codepoints, so zero term overlap.
2. **Silently wrong text.** The Word and PowerPoint exports report 0%
   presentation forms and look clean by every structural measure, but the text
   is wrong: `رؤيتنا` ("our vision") extracts as `وؤيتنا`, with letters
   substituted and dropped. These are valid Arabic codepoints that are not
   real words, so nothing structural can flag them. Verified by rasterising
   pages and reading them against the text layer.
3. **Low text density.** The PowerPoint and orientation documents yield 200 to
   300 characters per page, meaning most of their content is graphics and
   needs OCR regardless of whether the extracted text is sound.

### The failure signature depends on the extraction backend

All five documents, read with pdfplumber and with poppler's `pdftotext`:

| Document | pdfplumber chars | pdftotext chars | Presentation forms | pdftotext bidi controls /1k |
|---|---|---|---|---|
| Student Guide 2025 | 109,280 | 120,157 | 77% | 93.5 |
| Academic weeks 1448 | 4,811 | 5,776 | 71% | 156.5 |
| Guidance manual | 10,629 | 12,976 | 0% | 126.1 |
| Library services | 4,924 | 5,624 | 0% | 121.3 |
| Orientation 1446 | 2,595 | 3,089 | 0% | 139.9 |

The presentation-form ratio agrees between these two, but a third extractor, pypdfium2, returns base letters for the same glyphs, so the ratio describes how an extractor maps glyphs rather than the PDF alone. Word order differs too: pdfplumber writes Arabic in visual order, spelling every word backwards, which is why the pipeline extracts text with pypdfium2 ([calibration](eval/results/gate_calibration.md)).

`pdftotext` also writes a form feed after every page. Once those and its bidi marks are subtracted, the two backends agree on text length within 2% on four documents and 6.5% on the guidance manual. Quality gate thresholds are therefore only meaningful per backend, and must be recorded with one.

### Provenance

The 1448 academic calendar first added to the corpus turned out, on reading its footer, to be an unofficial student redesign that states it does not represent the Royal Commission. It was replaced with the official calendar. The unofficial file was also the only source of real control-character corruption seen so far, and of a large unexplained disagreement in character counts between the two backends. Both left the corpus with it.

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

A route is a prior, not a verdict. The quality gate still checks every
document, so a wrong prediction costs time and never correctness.

The rules come from five documents, and each rule in the code names the
documents it rests on. Of the three Microsoft documents, only the guidance
manual has been checked against its rendered page; the library guide's text
visibly scrambles, and the orientation guide's route is still a prediction.
Software with no evidence behind it, including other Microsoft and Adobe
products, falls to a default: try the text layer and let the gate decide.

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

The router predicts each route from metadata alone; the gate reaches the same
conclusion for all five documents by reading every word.

The gate still trusts two visibly broken pages, because their fragments are
real entries in a web word list. That limitation, the evidence behind every
threshold, and how to reproduce each number are in
[`eval/results/gate_calibration.md`](eval/results/gate_calibration.md).

## Usage

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
daleel inventory data/raw/            # aligned table
daleel inventory data/raw/ --json     # machine-readable
daleel route data/raw/            # predicted extraction path per document
daleel route data/raw/ --json
python3 scripts/fetch_lexicon.py   # the gate's word list: 69 MB, not committed
daleel gate data/raw/              # verdicts per document
daleel gate data/raw/ --pages      # verdicts per page, with reasons
```

## Dependency note

Text extraction uses **pdfplumber** (MIT) rather than PyMuPDF. PyMuPDF is
faster and exposes the same character geometry, but it is (AGPL) licensed, which
is a blanket disqualifier at many organisations and would force this
repository's own licence to match. On a 137 page corpus the speed difference is
irrelevant, and pdfplumber exposes geometry at character rather than word
level which the table reconstruction work needs anyway.

## Results

Extraction inventory above (table T1). Retrieval, OCR and arm-comparison
tables land here as that work completes. Nothing is published here before it
is measured.

## Licence

MIT, see [LICENSE](LICENSE).

## Source documents

The source documents are internal college publications and are not
redistributed here. See [`data/README.md`](data/README.md) for how to obtain
them.
