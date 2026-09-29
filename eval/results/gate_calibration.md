# Quality gate calibration

The quality gate decides, page by page, whether a PDF's text layer can be
trusted or the page must go to OCR. It has to decide without ground truth,
because almost no page will ever have any, so it works from proxies: numbers
computed from the extracted text alone. This document records how those
proxies were checked against hand-transcribed ground truth, which decisions
followed, and what the gate still gets wrong.

Every number in sections 1 to 3, in the lexicon note and in the count of pages
with letters from other scripts is printed by `scripts/calibrate_gate.py`. The
verdict tables in sections 5 and 6 are printed by `daleel gate data/raw/`, and
the page structures and word counts in section 6 by `scripts/inspect_pages.py`.
Ground truth stays on the annotator's machine, so the calibration script runs
locally and only its numbers are published here.

## Setup

- **Ground truth:** five pages, one per document, each transcribed by hand
  under [`eval/ANNOTATION.md`](../ANNOTATION.md), checked against a second
  transcription made independently, and reconciled against the page. By precision against that ground truth, two
  pages have sound text layers and three have broken ones.
- **Lexicon:** the CAMeL Lab's Modern Standard Arabic frequency list
  (Khalifa et al., 2021), CC BY-SA 4.0, fetched by `scripts/fetch_lexicon.py`
  against a pinned checksum.
- **Versions:** measured on two machines, with pypdfium2 5.6.0, pdfplumber
  0.11.9 and pdftotext 24.02.0 on one, and pypdfium2 5.13.0, pdfplumber
  0.11.10 and pdftotext 26.08.0 on the other. Everything the gate depends on,
  sections 2 and 3 and the lexicon note, came out identical on both. Parts of
  the extractor comparison in section 1 did not, and are reported with their
  versions.
- **Truth metrics:** *precision* is the share of the text layer's Arabic words
  that appear in the ground truth; *recall* is the share of the ground truth's
  words that the layer captured. Both compare bags of normalized words, so
  reading order does not affect them.

## 1. Choosing the extractor

The plan assumed pdfplumber, the tool the inventory uses. Measuring it against
ground truth showed it cannot serve as the text backend for Arabic.

| Extractor | Arabic words, guide p30 | Arabic words, calendar p01 | Digit and Latin runs reversed | Bracketed runs exact |
|---|---|---|---|---|
| pdfplumber | 5% / 5% | 24% / 24% | 0 of 11, 0 of 88 | 7/7, 5/5 |
| pdfplumber, `char_dir_render="rtl"` | 97% / 98% | 97% / 94% | 8 of 11, 87 of 88 | 0/7, 0/5 |
| **pypdfium2** | **100% / 99%** | **99% / 99%** | **0 of 11, 0 of 88** | **5/7, 5/5 on 5.6.0; 4/7, 3/5 on 5.13.0** |
| pdftotext | 99% / 99% on 24.02; 88% / 93% on 26.08 | 100% / 100% | 0 of 11, 0 of 88 | 0/7, 1/5 |

*Arabic words are precision / recall. Runs and brackets are counted on the two
sound pages, guide page 30 and the calendar.*

- **pdfplumber writes Arabic in visual order.** It sorts characters left to
  right, which spells every Arabic word backwards: الطالب comes out as بلاطلا.
  Digits, which run left to right anyway, come out correctly.
- **Its right-to-left option fails the other way.** Arabic words come out
  right and numbers come out reversed: `2025` becomes `5202`. It scored 97% on
  Arabic words while reversing 87 of the calendar's 88 digit and Latin runs,
  its dates among them, which no word-level metric would notice.
- **pypdfium2 gets both right,** on both builds tested. Its bracket placement
  is less stable: it kept most brackets around numbers in place, as in `(20%)`,
  but how many varied between PDFium builds. pdftotext detaches brackets, as in
  `)(20%`, and its accuracy on the guide page changed between poppler
  versions, from 99% to 88% precision.
- **Presentation forms depend on the extractor, not only the PDF.** pdfplumber
  and pdftotext return 77% and 71% presentation forms across the guide and the
  calendar (the inventory table in the README); pypdfium2 returns 0%, mapping those glyphs to base letters itself.
  An earlier version of the README said the ratio was a property of the PDFs
  because two extractors agreed; a third disagrees.

**Decision:** pypdfium2 extracts text (`daleel.ingest.extract`); pdfplumber
remains the tool for character geometry. Stability counts too: pypdfium2's
words and digits were identical across two PDFium builds, while pdftotext's
changed between poppler versions.

**Known limitation of pypdfium2:** it maps the calendar's "fi" ligature glyph
to U+001F, a control character, which normalization then removes, so
"finalizing" becomes "nalizing" and "Alfitr" becomes "Altr". pdfplumber reads
both correctly. The damage is visible to the gate's control-character count
and affects only English text; it is the one run pypdfium2 misses on the
calendar. `RCJY.gov.sa` is missing from guide page 30 in every extractor, so it
is absent from that page's text layer rather than lost by any of them.

## 2. Proxies against truth

With text from pypdfium2:

| Page | Route | Content chars | Arabic words (layer / truth) | Single letters (layer / truth) | Precision | Recall |
|---|---|---|---|---|---|---|
| guidance_manual p02 | ocr | 146 | 73 / 34 | 55% / 0% | 5% | 12% |
| student_guide_2025 p30 | text_layer | 1,467 | 290 / 292 | 1% / 1% | 100% | 99% |
| library_services p10 | ocr | 280 | 52 / 44 | 15% / 5% | 50% | 59% |
| orientation_1446 p03 | ocr | 331 | 85 / 70 | 15% / 0% | 33% | 40% |
| academic_weeks_1448 p01 | text_layer | 1,443 | 143 / 144 | 24% / 24% | 99% | 99% |

- **Normalization recovers both presentation-form documents** once the words
  arrive in the right order: 99 to 100% precision and recall.
- **The router's prediction for the orientation guide is confirmed.** Its
  route came from producer metadata alone; ground truth shows its text layer
  matches the page at 33% precision even through the best extractor.
- **Single-letter share does not separate sound from broken.** The calendar's
  legitimate single letters, the era suffixes on its dates and its grade codes,
  score 24%, above two of the broken pages.

## 3. The lexicon cutoff

Token validity is the share of a page's Arabic words found in the lexicon.
The cutoff decides which words the lexicon keeps.

| Page | Layer at 10,000 | at 1,000 | at 100 | Truth at 10,000 | at 1,000 | at 100 |
|---|---|---|---|---|---|---|
| guidance_manual p02 | 85% | 88% | 96% | 100% | 100% | 100% |
| student_guide_2025 p30 | 94% | 100% | 100% | 95% | 100% | 100% |
| library_services p10 | 75% | 81% | 88% | 95% | 95% | 95% |
| orientation_1446 p03 | 64% | 78% | 87% | 97% | 100% | 100% |
| academic_weeks_1448 p01 | 99% | 99% | 99% | 100% | 100% | 100% |

| Cutoff | Sound pages at least | Broken pages at most | Gap |
|---|---|---|---|
| 10,000 | 94% | 85% | +10 |
| **1,000** | **99%** | **88%** | **+12** |
| 100 | 99% | 96% | +3 |

At 10,000 the lexicon loses real but rarer words, so the guide dips; at 100 it
admits so much noise that the guidance manual's broken text scores 96%.
**Decision:** cutoff 1,000.

**Why a cutoff at all:** the list is built from web text, and web text
contains badly extracted Arabic PDFs. The broken word وؤيتنا, which the
guidance manual's layer produces for رؤيتنا, appears in the list 9 times; the
correct word appears 24,776 times. مصادرالمعلومات, printed run together on the
library slide, appears 38 times. Without a cutoff, the lexicon would certify
the very errors the gate exists to catch.

## 4. The rules

Each rule records the kind of evidence behind it: *measured* against ground
truth, *principle* (true by definition), or *judgement* (informed by the
evidence but not derived from it).

| Rule | Rejects a page when | Kind | Evidence |
|---|---|---|---|
| `token_validity` | fewer than 95% of its Arabic words are in the lexicon at cutoff 1,000 | measured | sound pages 99% or more, broken 88% or less |
| `too_few_words` | it has 1 to 19 Arabic words | judgement | validity over a handful of words says little; every calibration page had 52 or more |
| `cid_placeholders` | any `(cid:N)` placeholder appears | principle | each is a glyph the extractor could not read; none occur in the corpus |
| `replacement_characters` | any U+FFFD appears | principle | each is a character that could not be decoded; none occur in the corpus |
| `no_arabic_words` | it has no Arabic words at all | principle | there is nothing to judge |

**Why 95% and not the gap's midpoint:** the two mistakes cost different
amounts. Trusting a broken page puts wrong or missing text in front of a
student; rejecting a sound page costs one OCR run. The threshold leans toward
rejection.

## 5. The corpus at calibration

The five documents the calibration was built on, through `daleel gate data/raw/`
after all five rules:

| Document | Route | Pages | Trusted | Untrusted | No Arabic text | Gate says | Agrees |
|---|---|---|---|---|---|---|---|
| academic_weeks_1448 | text_layer | 3 | 3 | 0 | 0 | text_layer | yes |
| guidance_manual | ocr | 19 | 1 | 18 | 0 | ocr | yes |
| library_services_2024_2025 | ocr | 16 | 1 | 15 | 0 | ocr | yes |
| orientation_1446 | ocr | 13 | 1 | 12 | 0 | ocr | yes |
| student_guide_2025 | text_layer | 86 | 82 | 2 | 2 | text_layer | yes |

- **The router and the gate agree on all five documents.** The router predicts
  from metadata alone; the gate reaches the same conclusion by reading every
  word on all 137 pages.
- **The `too_few_words` rule came from this run.** Before it, guidance manual
  page 18 was trusted on its 3-word footer while its content was missing from
  the text layer. The rule changed three verdicts: it fixed that page, and it
  rejected two sound title pages of the guide, pages 2 and 85, whose 9 words
  are complete. Rejecting those costs an OCR run each.
- The guide's covers, pages 1 and 86, have no Arabic text layer at all.

## 6. Out of sample: four more documents

Four documents joined the corpus after the rules and the threshold were fixed:
the organizational regulations, the code of student conduct, the student
charter and the student portal guide. None of them contributed to any rule, so
they test the router and the gate on documents neither was built from. Their
sources are in `data/README.md`, and their checksums in `data/SHA256SUMS`.

| Document | Producer | Pages | Route | Trusted | Untrusted | No Arabic text |
|---|---|---|---|---|---|---|
| organizational_regulations | none | 58 | text_layer (default) | 43 | 1 | 14 |
| student_conduct_code | Adobe PDF library 15.00 | 12 | text_layer | 9 | 0 | 3 |
| student_charter | Adobe PDF library 15.00 | 8 | text_layer | 3 | 0 | 5 |
| student_portal_guide | GPL Ghostscript 10.00.0 | 42 | text_layer (default) | 37 | 4 | 1 |

- **The router held.** The conduct code and the charter use library version
  15.00, which no rule was built from, and routed correctly: pdfplumber shows
  75% and 76% presentation forms, and the gate trusts every one of their pages
  that has Arabic text. The regulations' download carries no metadata at all,
  and an online compressor rewrote the portal guide's producer and creator, so
  both took the default route. The gate then trusted their text layers.
- **The first apparent disagreement was the summary's fault.** The document
  summary first reported the charter as needing OCR, because five of its eight
  pages have no text layer. Those five are its cover, a photograph, two title
  pages and its back cover; all of its content is on the three trusted pages.
  The summary now judges a document's text layer only on pages with Arabic in
  it, and agrees with the router on all nine documents.
- **Text drawn as shapes.** Most pages without a text layer are outlines: ten
  of the regulations' fourteen carry between 1,100 and 4,750 vector curves and
  at most a few hundred characters, none of them Arabic letters. Among them is
  the page defining every grade code, which only OCR can read.
- **pdfplumber's tolerances assume standard pages.** The regulations, the
  conduct code and the charter define their pages at one tenth of A4.
  pdfplumber groups characters into lines and words with fixed tolerances, so
  on these pages it merges lines and splits words: regulations page 4 gives
  1,690 words through pdfplumber and 359 through pypdfium2, from the same 1,846
  letters. With its tolerances scaled to the page, pdfplumber gives 360.
  pypdfium2 needed no adjustment.
- **Glyph codes read as characters.** Regulations page 44 embeds a medical test
  form whose fonts carry no usable character map, so its text layer holds 769
  letters from Ethiopic and Canadian Syllabics instead of Arabic. The gate
  rejects the page only because it holds no Arabic words: the
  `cid_placeholders` rule never fires with pypdfium2, which writes wrong
  characters where pdfminer writes placeholders. It is the only one of the 257
  pages with such letters.
- **Two sound pages rejected.** Portal pages 8 and 18 fail validity at 93% and
  94%, and their only unknown words are real ones: the IBAN (الايبان), five
  times, and "passed" (المجتازة) as in passed courses, twice. The second is too
  rare in web text to pass the cutoff. The first is split across several
  spellings that normalization merges, and the lexicon applies its cutoff to
  each spelling before merging them; a separate change corrects that. Both
  errors are rejections, the direction the threshold was set to lean.

## 7. Limitations

- **Five calibration pages.** Two sound and three broken pages set the
  threshold. More ground truth would narrow the gap and could move it.
- **Fragments pass the lexicon.** Guidance manual page 11 (98% valid) and
  library page 16 (100% valid) are visibly broken into fragments, but the
  fragments are real entries in a web word list, for the same reason وؤيتنا
  is. They are still trusted. No rule is added for them: the obvious candidate,
  single-letter share, falsely accuses the calendar, and a rule without evidence
  would add ways to be wrong.
- **Word count cannot tell a sparse page from an incomplete one.** A title
  page with 9 correct words and a content page whose text is missing look alike
  to every text metric; the difference is only in the image. Checking
  completeness needs OCR output to compare against.
- **Rare real words cost pages.** A page that repeats a real word too rare
  for the lexicon can fall below the threshold, as two portal pages did. The
  cost is an OCR run, not a wrong answer.
- **Glyph codes read as characters are caught only when nothing else is
  there.** A page mixing sound Arabic with such letters would be trusted, its
  garbled words silently missing. None exists in the corpus today, and a rule
  for them is planned.

## Reproduce

```bash
python3 scripts/fetch_lexicon.py
python3 scripts/calibrate_gate.py
daleel gate data/raw/
```

The first needs network access; the second needs the ground truth under
`data/interim/ground_truth/`.
