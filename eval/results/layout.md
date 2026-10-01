# Layout: the library deck

The library deck (`library_services_2024_2025`, 16 slides exported from
PowerPoint 2019) was the first document taken whole through `daleel extract`.
Its text layer substitutes characters, so the gate sends every slide to OCR,
and each record holds dots.mocr's layout of its slide: blocks, each with a
category, a box and its text. This document records how those records were
checked against the slides, what held, what was changed, and what is left as
it is.

## Setup

- **Run:** dots.mocr at revision e539fbb5 on vLLM 0.30.0 with FP8 weights, as
  in [`ocr.md`](ocr.md): slides rendered at 200 DPI, 4000 by 2250 pixels, which
  the model's processor scales to 3080 by 1736 under a limit of 5.4 million
  pixels.
- **Checks:** every block's box drawn on its slide, rendered at 72 DPI so that
  a point is a pixel; each record's text read against its slide; the Arabic
  words of each record counted against its text layer's, the way the gate
  counts them; slide 10 compared with its reading in the T2 run.
- **Reading a block again:** 11 blocks read again two ways, on the same server:
  the whole slide with the block's box in dots.mocr's grounding prompt
  (`prompt_grounding_ocr`), and the block cut out of the slide at 200 DPI with
  its prompt for plain text (`prompt_ocr`). The script stayed out of the
  repository, and its answers stayed on the machine that ran it, like the rest
  of the OCR output.

## 1. What held

- **Boxes.** Every box lands on its block when drawn on its slide. Records keep
  boxes in the page's points, and the conversion from the model's resized image
  holds on all 16 slides.
- **Repeatability.** Slide 10 came back word for word as in the T2 run, under
  the same tag, so its normalized CER against the ground truth is still 2.1%.
- **Running text.** Outside titles, 97.8% of the 677 Arabic words in the final
  records are in the lexicon the gate uses, the text read from pictures
  included. In titles, 78.9% are (section 5).
- **Tables.** The table of contents (slide 2), the lending periods (8) and the
  study rooms (13) came back as HTML tables with every cell in place.

## 2. Reading order

- **dots.mocr reads blocks side by side left to right.** Slide 8 has two
  columns: on the right the slide's title, سياسة الاعارة, its first part and a
  table; on the left its second part, ثانيًا, and seven rules. The model read
  the left column first. On the academic calendar's page (T2) it read each row
  of event cards from the left, which puts every row's dates in reverse.
- **Swapping until nothing was left to swap made the calendar worse.** Every
  run read before a run beside it on its right was swapped, again and again.
  That fixed slide 8, but once a row of cards was fixed, its left card sat
  above the next row's left card, the two read as a column, and the calendar
  came out column by column.
- **Ordering by geometry alone broke what the model had right.** A recursive
  XY-cut at the widest gap fixed slide 8, but on the student guide's page 30
  (T2) it read two labels on the right before the values beside them, where the
  model had read each label with its value.
- **The rule kept** splits the model's own order where it moves on to another
  part of the page, between columns or between rows, at the widest gap. Two
  parts the model read left to right, side by side, are read right to left
  when both hold text (`daleel.ocr.reading_order`). Slide 8 and the calendar
  change, each calendar row now in the order of its dates. The other 15 slides
  and the 8 pages of five other documents read for T2 keep the model's order
  of text exactly.

## 3. Text inside pictures

- **Two slides lost their content.** On slide 3 the membership features are a
  mind map, and on slide 14 the booking steps sit in three boxes beside a QR
  code. dots.mocr called each region a picture, and a layout leaves a picture's
  text out (`LAYOUT_PROMPT`, rule 3), so each record held the slide's title and
  nothing more.
- **The loss shows in a count.** Slide 3's record held 7 Arabic words to the 40
  in its text layer, and slide 14's 8 to 25: 17.5% and 32%. Every other slide
  held at least 74% as many as its text layer, slide 12 the fewest at 74.4%.
  The text layer is too broken to read from, but its words can still be
  counted.
- **A block cannot be read again on its own:**

  | Block | Whole slide, the block's box in the prompt | Block cut out, plain text |
  |---|---|---|
  | Slide 3, the mind map | the slide's text, mind map included | the mind map, complete |
  | Slide 14, steps beside a QR code | the slide's text, steps included | nothing |
  | Slide 7, a photo | the slide's text | nothing |
  | Slide 4, a QR code | the slide's text | nothing |
  | Titles of slides 1, 3, 5, 10 and 16 | the slide's text, the title misread again | nothing on four; on slide 5 one word, repeated for 15,167 tokens and 621.6 s until the context ran out |
  | Slide 4, a line with the Hebrew letter | the slide's text, the letter now one the model could not write | nothing |
  | Slide 8, a rule with the Thai letter | most of the slide, the Thai letter again | the rule, the Thai letter again |

  dots.mocr ignored the box every time and read the slide, and a block cut out
  mostly came back empty. The grounding answers did show one thing: asked for
  text rather than a layout, the model reads the text inside pictures.
- **The fix** reads a page again, whole, with the prompt for plain text, when
  its OCR text holds less than half of the Arabic words in its text layer. The
  paragraphs no block holds become a block of their own after the page's
  largest picture (`daleel.ingest.records`). The answer is capped at 2,048
  tokens, after the runaway above. On this deck it read slides 3 and 14 again,
  in 19.7 and 16.9 s, and recovered all seven items of the mind map and all
  three booking steps. The other 14 slides were not read again.

## 4. Letters from other scripts

- **Two spellings come back in other scripts, the same way each time.** أسبوع,
  a week, comes back as أสوع, with the Thai letter SO SUA for سب: on slide 8,
  and three times on the organizational regulations' pages 21 and 33 in T2. ت
  came back once as the Hebrew ת, on slide 4: the same character with its first
  UTF-8 byte one lower.
- **Reading again does not mend them** (section 3): the Thai letter came back
  both ways, and the Hebrew one as a character the model could not write.
- **The fix** repairs both where they sit next to an Arabic letter, and notes
  every repair in the record (`daleel.ocr.letters`). No page in this corpus
  prints Thai or Hebrew. Any other letter of another script inside an Arabic
  word is left as read and noted. Whether the FP8 weights cause these slips is
  not known: the model was not served at full precision to compare.

## 5. Titles, left as read

- **Ten of the 16 slide titles are misread.** They are set in a decorative
  font, and dots.mocr reads مميزات as ميزان, المكتبات as المتبان, المكتبة as
  الملكية, المتاحة as النامة, and the cover's title, دليل الطالب للاستفادة من
  خدمات المكتبات, as ريل الطالب للاستفارة من خمسان الكبار. One more title loses
  only its taa marbuta, which normalization folds.
- **Neither way of reading a title again helps** (section 3).
- **Nine of the ten are in the deck correctly too.** The table of contents on
  slide 2, in a plain font, lists them with their slide numbers; only the
  cover's title is nowhere else. A chunker could take its section titles from
  there. That waits until retrieval shows the titles matter, since the text
  under them is read well.

## 6. Speed

- **16 slides in 456 s**, 17.3 to 49.6 s a slide. The two pages read again for
  their text took 36.6 s more.
- **A reading's time is two rates.** The 22 timed requests of section 3 fit,
  each within 2.5 s, about 520 prompt tokens a second and 24.5 written tokens a
  second. A slide at 5.4 million pixels is 6,875 prompt tokens, so about 13 s
  pass before the model writes anything.
- **So a slide's fixed cost is its image.** The DPI sweep will show what fewer
  pixels cost in accuracy, and so what they would save.

## 7. Limits

- One document of 16 slides, read once. The rules for reading order and for
  reading a page again were set on it and checked on 9 more pages from T2, not
  on pages held out for the purpose.
- The thresholds rest on this deck: half of the text layer's words, at least 10
  of them, and 80% of a paragraph's words for a block to hold it. The lowest
  slide left alone is at 74.4%, and the highest read again at 32%.
- A page with no text layer, or with fewer than 10 Arabic words in it, is never
  read again, so text inside its pictures stays missing.
- Text recovered from a picture carries the picture's box, not its own: a
  reading for text alone gives no boxes.
