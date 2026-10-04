# OCR engines (T2)

The quality gate sends a page to OCR when its text layer cannot be trusted.
This document compares three OCR engines on the pages that have hand-made
ground truth: Tesseract, the long-standing open-source baseline; PaddleOCR, a
current detector and recognizer pair; and dots.mocr, a vision language model
that reads a whole page and answers with its layout. It records how each was
set up, what each gets wrong, and the choices made on the way.

The T2 table comes from `scripts/ocr_eval.py`: its accuracy columns from
`scripts/ocr_eval.py saved`, which scores each run's saved output, so every
engine is scored by the same code against the same ground truth, and its
seconds from the timed runs. The error analyses in sections 1 and 2 were made
on the same saved output and, for dots.mocr, on the server's full responses,
which record each page's tokens. Ground truth and OCR output stay on the
annotator's machine, so only their scores are published here.

## Setup

- **Ground truth:** 10 pages from 7 documents, in 11 pieces: 4 whole pages,
  3 regions of pages, 3 tables and 1 page's text around its table. Each was
  transcribed by hand under [`eval/ANNOTATION.md`](../ANNOTATION.md), then
  checked against a second transcription made independently by a language
  model from the page image; every difference was resolved by looking at the
  page (ANNOTATION.md 8). Pages and regions hold 5,763 characters; all pieces
  together, 1,564 words.
- **Scores:**
  - *CER raw* and *CER normalized*: character edits per character of ground
    truth, on pages and regions, the normalized form after the normalization
    retrieval uses, which also writes Arabic-Indic digits and Arabic
    punctuation in their ASCII forms.
  - *WER normalized*: the same over words.
  - *Words missed*: the share of the ground truth's words found nowhere in the
    engine's output, as a bag, on every piece. Tables are scored by words
    rather than by edits, since their cells have no single reading order.
  - *Words added*: the share of the engine's words found nowhere in the ground
    truth, as a bag, on the 5 pages whose ground truth covers the whole page,
    one of them the calendar. Section 2 explains why this column exists.
  - Rates are summed over pieces before dividing, so each piece counts by its
    length.
- **Rendering:** pypdfium2 at the page's printed size; 300 DPI for Tesseract
  and PaddleOCR, 200 DPI for dots.mocr, the resolution its own parser uses.
- **Timing:** each engine reads the first page once untimed, then every page
  once; *s / page* is the mean of those ten reads.
- **Machine:** Intel Core i5-12400F (6 cores, 12 threads) and an NVIDIA
  GeForce RTX 3060 Ti with 8 GB, used by dots.mocr only; Windows with WSL 2,
  Python 3.12.14, pypdfium2 5.13.0.
- **Engines:**
  - Tesseract 5.5.0 with the `tessdata_best` models (`ara` sha256 ab9d157d,
    `eng` 8280aed0), page segmentation mode 3.
  - PaddleOCR 3.7.0 with PaddleX 3.7.2, PaddlePaddle 3.2.2 and python-bidi
    0.6.11; the PP-OCRv5 server detector (sha256 b3e98695) and Arabic mobile
    recognizer (881acf8b); on the CPU, 10 threads, MKL-DNN; lines scored under
    0.5 dropped.
  - dots.mocr (rednote-hilab) at revision e539fbb52280, served by vLLM 0.30.0;
    FP8 weights through the Marlin kernel; pages of up to 5.4 million pixels;
    the layout prompt, temperature 0.1, top_p 1, seed 0; a context of 16,384
    tokens. The fixed seed makes it repeatable: two runs gave identical text on
    every page.

## T2

| Engine | CER raw | CER norm | WER norm | Words missed | Words added | s / page | Layout-aware |
|---|---|---|---|---|---|---|---|
| Tesseract, `ara` | 12.5% | 11.7% | 16.6% | 24.6% | 32.2% | 1.7 | no |
| Tesseract, `ara+eng` | 15.0% | 14.4% | 19.5% | 15.7% | 15.5% | 2.8 | no |
| PaddleOCR, min score 0.5 | 12.3% | 12.1% | 19.5% | 7.9% | 14.3% | 11.1 | line boxes |
| **dots.mocr, FP8** | **4.4%** | **4.4%** | **5.6%** | **4.3%** | 22.1% | 120.6 to 264.4 | **yes** |

*Layout-aware:* whether the output keeps the page's structure. Tesseract gives
plain text. PaddleOCR gives each line's box, which is how this project orders
its lines. dots.mocr gives every block's box and category (heading, list item,
table, page header, picture) and each table as HTML.

- **dots.mocr reads text about three times as accurately as either classical
  engine,** and misses half as many words as PaddleOCR. On running text it is
  close to perfect: 3 word errors in the 486 words of the three regulation and
  charter regions, against 26 for Tesseract and 46 for PaddleOCR.
- **It also adds words the page never printed,** which no other column shows:
  on the academic calendar it invented a second date in every one of 14 date
  boxes. Section 2 has the detail. Its 22.1% of added words is more than
  PaddleOCR's 14.3% and less than Tesseract `ara`'s 32.2%, but the kinds
  differ. The classical engines' added words are misreadings, mostly garbled.
  Of dots.mocr's 211, 121 are the calendar's invented entries, believable and
  wrong; the rest are the logo (36), dates written differently (44) and a few
  misreadings (10).
- **It is slow, and its speed was not steady.** It needs a GPU, and two runs
  with identical output took 120.6 and 264.4 seconds a page: 71 and 156 times
  Tesseract `ara`'s 1.7 seconds, 11 and 24 times PaddleOCR's 11.1. The first
  run wrote 12 tokens a second overall, the second 5.5, with every page 1.3 to
  4.1 times slower than the first time. Within the first run the rate fell from
  37 tokens a second on the untimed first read, to 24 reading the same page
  again, to about 13 on the pages after. The same request slowing down points
  at the machine rather than the model; the cause was not found.
- **The logo counts against every engine, and most against the best one.**
  [ANNOTATION.md](../ANNOTATION.md) 5.2 leaves the Royal Commission's logo out
  of the ground truth, but it is readable, and dots.mocr reads it perfectly as
  a page header. That is every one of its 64 edits on the guidance page. With
  the logo taken out of its output, its CER falls from 4.4% to 1.3%.
- **Normalization forgives little, most of it Tesseract's.** Of Tesseract
  `ara`'s 720 character edits it forgives 48: 26 commas, because Tesseract
  writes the pages' Arabic commas as Latin ones, 10 yaa and alef maqsura
  swapped, 8 diacritics and 4 hamzas on alef. It forgives 12 of PaddleOCR's
  710 edits and 3 of dots.mocr's 256. Digits change no CER here: every
  Arabic-Indic digit in the ground truth is in a table, and tables are scored
  by words.

## 1. What each engine gets wrong

On the three running-text regions, 486 words: regulations pages 5 and 21 and
the student charter.

| Engine | Word errors | What they are |
|---|---|---|
| Tesseract, `ara` | 26 | short words clipped (في read as ي or ق, التي as الي); percentages read as `1670` and `9630` |
| PaddleOCR, min score 0.5 | 46 | short words clipped (التي read as الي, في as ف, لا as ل); text read into graphics |
| dots.mocr | 3 | أسبوع read as أสوع, with a Thai letter; لتؤكد as لتأكد; ويمثل as وبمثل |

- **Tesseract loses whole blocks on designed pages.** It skips text it does not
  recognize as text: 24.6% of all words missed, and 55.4% of the calendar's and
  64.2% of the grade table's. Reading `ara` alone, it writes English as
  Arabic-looking garbage. Adding `eng` recovers the English and halves the
  words added, 32.2% to 15.5%, at a cost to the Arabic: CER rises from 12.5%
  to 15.0%, and on the running-text regions from 2.9% to 4.1%.
- **PaddleOCR keeps the blocks but clips short words.** Its misses are a third
  of Tesseract's, yet on running text it makes more word errors, 46 to 26.
  Many are the shortest, most common words with a letter dropped: التي becomes
  الي, في becomes ف and لا becomes ل. It also reads text into background
  graphics, some of which the minimum score of 0.5 filters out (section 3).
- **dots.mocr reads nearly everything, and differs from the ground truth in
  three ways.** Without the logo, it has 28 word differences on pages and
  regions:
  - 14 are reading order, in designed headers, footers and numbered boxes, as
    on the orientation page.
  - 7 are the page's text rewritten the way it is commonly written, where the
    ground truth keeps what is printed (ANNOTATION.md 3.1): the run-together
    والصوروالكثيرمن مصادرالمعلومات gets its spaces, and لم يؤدِ, correct as
    printed, becomes the common but ungrammatical لم يؤدي.
  - 7 are its own misreads: the three in the table above; المكتبة as المتبة
    and السعودية as السعورية in the library slide's title; a dropped `X`; and
    المواظبة written as المواضبة, a common misspelling.
  - It also changes spellings that normalization hides: it corrects the page's
    خدمه to خدمة, and drops the hamza from إصدار.
- **One misread repeats.** Every time the word appears on the regulations
  pages read here, أسبوع twice on page 21 and أسبوعين once on page 33, it wrote
  a Thai letter in place of سب; the student guide's أسبوعاً came out right. Whether the FP8 weights cause this
  is not measured, since the unquantized model could not be run on this GPU
  (section 3).

## 2. The calendar: an invented date in every box

The academic calendar lays out 14 events as cards, each with its title in
Arabic and English, the day in Arabic and English, and the Gregorian and Hijri
dates. dots.mocr read each card's date strip as a two-cell table:

- **The first cell is right in all 14:** the Arabic day and both dates.
- **The second cell is invented in all 14.** In 11 it repeats the same dates
  under another weekday, always Sunday or Thursday, the first and last days of
  the Saudi working week: Sunday where the event is on a Thursday, Thursday
  otherwise. In the other 3 it copies a neighbouring event's day and dates.
- **All 14 English day names are gone,** apparently replaced by the invented
  cell.

So the model's text for the midterm exams says both Sunday 11 October 2026 and
Thursday 11 October 2026; the date is a Sunday. A question about the midterms
could be answered with the wrong day, cited to the right page.

**Words missed could not see it.** Missed words ask whether each word of the
ground truth appears somewhere in the output. The invented cells add words and
remove none, so they cost nothing there. dots.mocr's 15.0% missed on the
calendar is 18 English day-name words, 14 Gregorian dates written with the era
letter apart (`23 م` for `23م`), 14 Hijri dates without their `ه`, and one
word split in two. That is why the evaluation now also counts words added, on
every page whose ground truth covers the whole page. On the calendar dots.mocr
added 174 words: 121 from the invented cells, 9 from the logo, and 44 almost
all from dates written differently.

The classical engines fail differently. Tesseract `ara` adds more words on the
calendar, 58.3% against 37.7%, mostly garbled. Some are shaped like dates,
such as `1448/06/18-0ه` for `1448/06/18-10ه`, but each is a misreading of a
date printed there. Tesseract and PaddleOCR read the page one line image at a
time, so what they add is a misreading of something on the page. A language
model writes what is likely, and on a calendar a likely date can be a wrong
one.

This rests on one page, so how often it happens is not known. The two grid
tables, the grade scale and the fees table, came out with no invented cells.
What the calendar shows is the kind of error: the one most costly for a system
that answers students' questions about dates and deadlines.

## 3. Choices made on the way

- **Tesseract: the version and the model set matter as much as the engine.**
  Tesseract 5.5.0 with the fast, integer models drops whole lines of text on
  designed pages that 5.3.4 reads, with the same model file, image and scorer:
  CER 16.3% against 11.9%, and 44.7% against 9.7% on the library slide.
  Forcing its generic code path changed nothing, which rules out the CPU's
  vector instructions. With the float `tessdata_best` models 5.5.0 recovers,
  12.5% against 5.3.4's 11.4%, so T2 uses them.
  - A `TESSDATA_PREFIX` that does not exist makes Tesseract fall back to its
    own models with only a warning, which once made a run meant for the best
    models use the fast ones. `daleel.ocr.tesseract` now refuses such a run.
- **PaddlePaddle is pinned at 3.2.2.** Versions 3.3.0 and 3.3.1 crash on
  MKL-DNN, the default CPU path, and without MKL-DNN the server detector
  allocated 58 GB for the calendar at 300 DPI.
- **PaddleOCR drops lines it scores under 0.5.** It keeps everything by
  default, including what it reads in background graphics; PaddleOCR 2 dropped
  lines under 0.5 by default. The filter lowers CER from 13.2% to 12.3% and
  words added from 15.7% to 14.3%, and misses no more words.
- **Widening the page did not help PaddleOCR.** PaddleOCR issue 18349 proposes
  widening each line image before recognition, to give the recognizer more
  steps per letter. Stretching the whole page to 1.5 times its width
  approximates that. With the minimum score in place it raised CER from 12.3%
  to 12.8% and words missed from 7.9% to 9.4%, and took 42% longer. Without
  the minimum score it lowered CER a little, 13.2% to 12.8%, but still missed
  more words, 9.2%. The issue's own fix, per line and capped, is not tested.
- **PaddleOCR's lines are put in reading order by this project.** PaddleOCR
  sorts the boxes of a row left to right. `daleel.ocr.reading_order` puts two
  boxes in one row when each one's vertical middle lies within the other's
  height, anchors each row on its tallest box, and reads it right to left.
- **dots.mocr on an 8 GB card.** Its unquantized weights take 5.66 GiB of the
  6.96 GiB the Windows desktop leaves free, too little for the rest, so they
  are quantized to FP8 when the server loads them. This GPU has no FP8
  arithmetic, so the Marlin kernel keeps the weights in FP8 and computes in
  bf16. vLLM 0.30.0 picks a kernel that fails on this generation of GPU
  ([vllm#55008](https://github.com/vllm-project/vllm/issues/55008)), so the
  server is started with `--linear-backend marlin`.
  - vLLM reserves memory for the largest image the model accepts, 11.3 million
    pixels, which left no room for the KV cache. Pages are capped at 5.4
    million pixels instead, which keeps every page at 200 DPI except the
    library slide, read at about 155 DPI; its text is large. The engine sends
    the cap with every page, so the cap recorded is the one used.
  - dots.mocr's own client asks for up to 32,768 tokens per answer, or 16,384
    from its command line. vLLM refuses either on a 16,384-token context,
    since the page's image tokens must fit too, so the engine sends no limit
    and an answer may run to the end of the context. The longest request, the
    calendar, used 10,844 of the 16,384 tokens.

## 4. What this means for Daleel

- **For running text, dots.mocr is clearly the best reader here.** Its 3 word
  errors in 486 would barely touch retrieval, while the classical engines'
  clipped function words and misread numbers would.
- **Its tables need a check before they feed answers.** The two grid tables
  came out clean, but the calendar's date boxes, a table-like layout, came out
  with a wrong date in every one, and a wrong date in a cited answer is worse
  than a missing one. One check would keep only the dates a second engine also
  reads on the page; another would leave such pages to PaddleOCR.
- **Its cost is time.** Two to four and a half minutes a page on this GPU,
  against 11 seconds for PaddleOCR on the CPU. OCR runs once, when a document
  is ingested, so this is paid once per page rather than once per question.
  Reading the whole corpus later was faster: 80 pages in 48 minutes, 22
  seconds a page at the median and 139 at most (README, Extraction).

## 5. Limits

- Ten pages, and one of them a full table page: the calendar finding rests on
  that page, consistent across its 14 boxes.
- dots.mocr as served here is not the model as released: FP8 weights, and one
  page scaled below 200 DPI. The effect of either is not measured.
- The check transcriptions were made by a language model, so a model's habits,
  such as correcting a page's typos, had a chance to reach the ground truth.
  Every difference was resolved against the page, and the ground truth keeps
  the typos dots.mocr corrects.
- Words added can only be counted where the ground truth covers the whole
  page: 5 of the 10 pages.
- Timing is from one machine running WSL 2, with the GPU shared with the
  Windows desktop, and dots.mocr's speed more than halved between two runs of
  the same pages. Its seconds are a range, not a measurement of the model.

## 6. Every page

T2 combines these rows, each piece counting by its length. PaddleOCR is the run
with a minimum score of 0.5.

Pages and regions, CER normalized / WER normalized, in percent:

| Ground truth | Kind | Characters | Tesseract `ara` | Tesseract `ara+eng` | PaddleOCR | dots.mocr |
|---|---|---|---|---|---|---|
| Guidance manual, page 2 | page | 222 | 65.3 / 68.6 | 64.4 / 62.9 | 32.4 / 34.3 | 28.8 / 25.7 |
| Library services, page 10 | page | 329 | 9.7 / 23.4 | 18.5 / 29.8 | 4.3 / 21.3 | 2.1 / 14.9 |
| Regulations, page 5 | region | 466 | 3.0 / 8.3 | 3.0 / 8.3 | 6.7 / 16.7 | 0.0 / 0.0 |
| Regulations, page 21 | region | 1,795 | 2.0 / 5.4 | 3.6 / 6.4 | 5.3 / 9.0 | 0.5 / 0.3 |
| Orientation, page 3 | page | 433 | 14.1 / 12.7 | 13.4 / 15.5 | 18.7 / 22.5 | 16.4 / 18.3 |
| Student charter, page 4 | region | 665 | 1.1 / 2.9 | 2.6 / 3.9 | 2.3 / 5.9 | 0.3 / 2.0 |
| Student guide, page 30 | page | 1,853 | 20.5 / 28.2 | 25.4 / 34.6 | 21.2 / 32.4 | 5.4 / 6.7 |

Every piece, words missed / words added, in percent. Words added are counted
only on whole pages, the calendar's on its table's row.

| Ground truth | Kind | Tesseract `ara` | Tesseract `ara+eng` | PaddleOCR | dots.mocr |
|---|---|---|---|---|---|
| Calendar, page 1: the table | table | 55.4 / 58.3 | 15.3 / 21.8 | 11.5 / 13.0 | 15.0 / 37.7 |
| Calendar, page 1: text around the table | page text | 38.1 | 14.3 | 14.3 | 0.0 |
| Guidance manual, page 2 | page | 25.7 / 25.7 | 25.7 / 21.2 | 2.9 / 26.1 | 0.0 / 20.5 |
| Library services, page 10 | page | 19.1 / 20.8 | 14.9 / 16.7 | 17.0 / 18.8 | 8.5 / 14.0 |
| Regulations, page 5 | region | 1.4 | 1.4 | 4.2 | 0.0 |
| Regulations, page 21 | region | 3.5 | 4.5 | 5.4 | 0.3 |
| Regulations, page 27: grade table | table | 64.2 | 54.3 | 11.7 | 4.9 |
| Regulations, page 33: fees table | table | 2.6 | 6.0 | 10.3 | 0.9 |
| Orientation, page 3 | page | 12.7 / 0.0 | 12.7 / 3.1 | 4.2 / 18.1 | 0.0 / 11.2 |
| Student charter, page 4 | region | 2.9 | 3.9 | 2.9 | 2.0 |
| Student guide, page 30 | page | 17.3 / 8.8 | 17.9 / 9.5 | 6.1 / 12.5 | 1.3 / 3.8 |

## Reproducing

With the ground truth in `data/interim/ground_truth/` and the source PDFs in
`data/raw/`:

```bash
TESSDATA_PREFIX=/path/to/tessdata_best python3 scripts/ocr_eval.py tesseract
TESSDATA_PREFIX=/path/to/tessdata_best python3 scripts/ocr_eval.py tesseract --lang ara+eng
python3 scripts/ocr_eval.py paddle --min-score 0.5          # with the paddle extra installed
python3 scripts/ocr_eval.py dots --quantization fp8 --max-pixels 5400000
python3 scripts/ocr_eval.py saved data/interim/ocr/<run>    # score a saved run again
```

dots.mocr needs a vLLM server on the same machine, started as:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve rednote-hilab/dots.mocr --trust-remote-code \
  --chat-template-content-format string --quantization fp8 --linear-backend marlin \
  --max-num-seqs 1 --mm-processor-kwargs '{"max_pixels": 5400000}' \
  --gpu-memory-utilization 0.8 --max-model-len 16384
```

`VLLM_USE_FLASHINFER_SAMPLER=0` keeps vLLM from compiling a sampler it would
use only for its warm-up, which needs the CUDA compiler; requests with a seed
never use it.
