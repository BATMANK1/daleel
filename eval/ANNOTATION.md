# Ground truth annotation guidelines

Ground truth records what a careful human reader sees on a rendered page. Three
things are measured against it:

- the **extraction quality gate**: is a document's text layer trustworthy?
- **OCR engines**: character and word error rate
- **table reconstruction**: cell-by-cell accuracy

An error here silently corrupts every number downstream, so consistency matters
more than speed. Rules are numbered so a judgement call can cite the rule it
applied.

## Quick reference

1. Type what is **printed**, including the page's own mistakes.
2. Transcribe from the **page image only**, never from extracted text.
3. **No tatweel.** Diacritics **only if printed.** Digits **as printed.**
4. Read **right to left, top to bottom.** Keep the page's line breaks.
5. Headers, footers and page numbers **in**. Logos, images and watermarks **out**.
6. Tables go to **CSV**, everything else to `.txt`.
7. Can't read it? **`[?]`**, never a guess.
8. Finish your version **before** looking at any other transcription.

## 1. Files

1.1 One file per page, named `<doc_id>_p<NN>.txt` with a two-digit page number:
`guidance_manual_p02.txt`. `NN` is the PDF page index, as `pdftoppm` numbers
it, not the page number printed on the page.

1.2 A table on the page goes in a separate `<doc_id>_p<NN>.csv`. Everything else
on that page stays in the `.txt`. See section 6.

1.3 Files live in `data/interim/ground_truth/`, which is gitignored. They are
**never committed**: a transcription reproduces the document's text, and the
documents are not redistributable. Back the folder up privately, since git is
not tracking it.

1.4 UTF-8, LF line endings, no byte order mark, one trailing newline.

1.5 Dense pages may be transcribed by region rather than in full. A region is
one or more complete blocks (see 4.3), transcribed completely under the same
rules and named `<doc_id>_p<NN>_r<K>.txt`. The log records which blocks each
region covers.

## 2. Source

2.1 Transcribe from the rendered page image only: `data/interim/pages/`, 300 DPI.
Zoom as far as you need.

2.2 Never copy from the PDF's text layer, `pdftotext`, pdfplumber, OCR output, or
another person's or model's transcription. The ground truth exists to measure
those; copying from any of them makes it measure nothing.

## 3. Characters: record the page, not correct Arabic

3.1 Type exactly what is printed, **including the page's own spelling mistakes**.
If a page prints اجازة without a hamza, type اجازة. Normalization is measured
later against this file, so the file must keep exactly what normalization has
to fix.

3.2 Distinguish these exactly as printed: ة and ه, ى and ي, ا and أ and إ and آ,
ء and ؤ and ئ.

3.3 **Tatweel** (ـ, the stroke that stretches letters for justification) is never
typed. A printed الأحـــد is typed الأحد.

3.4 **Diacritics** (fatha, damma, kasra, shadda, sukun, tanween) only where they
are printed, each typed directly after the letter it sits on. Placement varies
even within one page: tanween may sit on the letter before the alef (يومًا) or
on the alef itself (أسبوعاً). Type what each word shows.

3.5 **Ligatures** are typed as their component letters. لا is ل followed by ا,
which the keyboard does for you. A single-glyph ﷲ is typed الله.

3.6 **Digits exactly as printed.** Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) stay
Arabic-Indic; Western digits (0123456789) stay Western. The same applies to
separators: ٫ versus `.`, and ، versus `,`.

3.7 **Punctuation as printed**: Arabic comma ، · Arabic semicolon ؛ · Arabic
question mark ؟ · and their Latin forms wherever the page uses those.

3.8 **Latin text** as printed, case preserved.

3.9 **Spaces**: one space between words. Visual gaps created by justification or
tatweel are not spaces. No spaces at the start or end of a line.

## 4. Reading order and layout

4.1 Right to left, top to bottom. When blocks sit side by side, finish the
right-hand block before starting the left.

4.2 In running text, keep the page's line breaks: one printed line is one line
in the file.

4.3 Separate blocks (heading, paragraph, box, caption, footer) are separated by
one blank line. Each heading goes on its own line.

4.4 Numbers inside Arabic text go where they occur in the sentence, with the
number itself typed normally: `3.75`. A run of Latin letters, numbers and
symbols (`X 20%`, `(DN)`) is read left to right as a single unit, even inside
Arabic text. The editor may display such runs in a strange position; trust the
order you typed, not the display. To check the order of items on a mixed line,
split its image into left and right halves: the right-hand item is read first.

4.5 **Ranges on an Arabic page** are typed in reading order, right-hand item
first. A range printed as `28/11/2026 - 20/11/2026` is typed
`20/11/2026 - 28/11/2026`. Sanity check: the first date should be the earlier
one.

4.6 **Brackets** are typed logically: opening bracket before the enclosed text,
closing bracket after. Editors mirror them in right-to-left display, which is
correct and not something to fix.

4.7 **List numbering** is typed if it is text (١. or 1- or أ). Decorative
bullets, check marks and icons are omitted.

4.8 A **number badge** attached to a box (01, 02, 03) belongs to that box: it
is typed on the line directly after the box's text, with no blank line
between them.

## 5. What to include

5.1 **Include**: headings, body text, captions, headers, footers and page
numbers.

5.2 **Exclude**: text that is part of a logo, photo or illustration (including
the Royal Commission logo in page headers, even though it is readable),
watermarks, and purely decorative elements.

5.3 Text inside a coloured box or banner is included. The line is drawn at
logos and pictures, not at colour.

## 6. Tables

6.1 A table (any grid of rows and columns, calendars included) is transcribed to
CSV, not text. A grid has no single reading order, so it is compared cell by
cell rather than character by character.

6.2 The first row is a header with column names you choose. One CSV row per table
row. Section 3 applies inside every cell.

6.3 A cell spanning several rows has its value repeated in each row it covers.

6.4 An empty cell is an empty field.

6.5 **Information carried only by colour or icons**, such as a highlighted day
matched to a legend, goes in an explicit column using the legend's exact
wording. Two items in one cell are separated by ` | `. That column is a human
reading of colour; record it as such in the page log (section 8).

6.6 A field containing a comma is wrapped in double quotes.

6.7 Both annotators **agree the columns before either starts**. Cell-by-cell
comparison only works if the columns are identical.

6.8 Text that wraps onto several lines within one cell is joined into one line
with single spaces. A CSV cell holds no line breaks.

6.9 Dates are typed as printed, including suffixes such as م, with هـ typed as
ه under 3.3. Ranges go start first under 4.5, including compressed ranges: a
range printed `2026/09/26-23` is typed `2026/09/23-26`. The same applies to
ranges of day names.

## 7. Uncertainty

7.1 An unreadable word is typed as `[?]`. Never guess: a guess becomes a false
fact that every measurement then inherits.

7.2 Torn between two readings? That is also `[?]`. Do not pick the more likely
one.

## 8. Checking

8.1 After finishing a page, reread the whole file against the image once, line
by line.

8.2 Then cross-check independently. Obtain a second transcription of the same
page, made without seeing yours, and store it as
`data/interim/ground_truth/check/<doc_id>_p<NN>.model.txt`. Compare word by
word:

```bash
git diff --no-index --word-diff \
  data/interim/ground_truth/check/<doc_id>_p<NN>.model.txt \
  data/interim/ground_truth/<doc_id>_p<NN>.txt
```

8.3 Resolve **every** difference by looking at the page. Change your file only
where the page shows you were wrong. Never paste the other version over yours.

8.4 Keep a short log per page in `data/interim/ground_truth/LOG.md`: how many
differences there were, how many were your errors and how many the other
transcription's, and any judgement calls made under sections 5 and 6.

## 9. Changing these rules

9.1 If a page forces a new rule, add it here and re-check the pages already done
against it. Consistency across pages matters more than any single rule being
perfect.
