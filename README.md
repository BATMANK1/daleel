# Daleel · دليل

[![CI](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml/badge.svg)](https://github.com/BATMANK1/daleel/actions/workflows/ci.yml)

Answers questions about Royal Commission for Jubail and Yanbu college
regulations in Arabic, with the exact clause cited.

> **Status: in development.** This README grows with the repository.
> No result is published here until it has been measured — see
> [Results](#results).

## The problem

Students need answers from 137 pages of Arabic regulation documents:
*Can I transfer majors? What GPA do I need? When is the drop/add deadline?*
Today the options are reading all of it, asking a classmate who may be wrong,
or waiting for the registrar.

An unsourced answer about your own academic standing is unusable — you cannot
act on it and the registrar cannot confirm it. So **every answer this system
gives carries a document, page and clause reference.** Citation is a hard
functional requirement, not a feature.

## Why this is not a "chat with your PDF" project

Measuring the corpus before building anything turned up extraction damage that
no naive pipeline would notice. `daleel inventory data/raw/` reports it:

| Document | Producer | Pages | Chars | ch/page | Presentation forms |
|---|---|---|---|---|---|
| Student Guide 2025 | PDFium | 86 | 109,280 | 1,271 | **77%** |
| Academic weeks 1448 | Illustrator | 3 | 15,060 | 5,020 | **61%** |
| Guidance manual | Word 2019 | 19 | 10,629 | 559 | 0% |
| Library services | PowerPoint 2019 | 16 | 4,924 | 308 | 0% |
| Orientation 1446 | Word 2019 | 13 | 2,595 | 200 | 0% |

Three distinct failure modes sit behind those numbers:

1. **Presentation-form substitution.** In the two design-tool exports, most
   Arabic is stored as Unicode Arabic Presentation Forms -- the contextual
   glyph variants a renderer picks per letter position -- rather than as base
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

Worth stating because it changes how the gate must be calibrated. The same two
documents, read with poppler's `pdftotext` instead of pdfplumber:

| Document | Backend | Chars | Presentation forms | Bidi controls /1k | C0 controls /1k |
|---|---|---|---|---|---|
| Academic weeks 1448 | pdftotext | 8,483 | 61% | 232.0 | 125.8 |
| Academic weeks 1448 | pdfplumber | 15,060 | 61% | 0.0 | 0.0 |
| Guidance manual | pdftotext | 12,975 | 0% | 126.1 | 1.5 |
| Guidance manual | pdfplumber | 10,629 | 0% | 0.0 | 0.0 |

Presentation-form ratio agrees across backends, so it is a property of the PDF.
The bidirectional and C0 control characters are not: `pdftotext` inserts bidi
marks itself to make RTL output display correctly, and pdfplumber emits none.
An earlier draft of this README reported those control characters as corruption
in the source documents. They are largely an artefact of the reading tool.

Two consequences. Gate thresholds are only meaningful per backend and must be
recorded with one. And the character-count disagreement -- pdfplumber returning
nearly twice as much text for the calendar -- is itself unexplained and flagged
for the table-reconstruction work, since duplicated or overlapping text objects
would produce exactly this.

Every naive pipeline built on this corpus fails, and fails *silently*:
retrieving nothing relevant and generating a confident answer anyway. So
extraction here is a component with its own tests and its own quality gate,
rather than a preprocessing line.

## Usage

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
daleel inventory data/raw/            # aligned table
daleel inventory data/raw/ --json     # machine-readable
```

## Dependency note

Text extraction uses **pdfplumber** (MIT) rather than PyMuPDF. PyMuPDF is
faster and exposes the same character geometry, but it is AGPL-licensed, which
is a blanket disqualifier at many organisations and would force this
repository's own licence to match. On a 137-page corpus the speed difference is
irrelevant, and pdfplumber exposes geometry at character rather than word
level -- which the table reconstruction work needs anyway.

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
