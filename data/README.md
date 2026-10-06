# Source documents

The PDFs this project reads are RCJY college publications. They are **not
committed to this repository**, and `data/raw/` is gitignored.

To run the pipeline, place these nine files in `data/raw/`:

| Expected filename | Document | Pages | Source |
|---|---|---|---|
| `student_guide_2025.pdf` | Student Guide, RCJY colleges and institutes, 2025 | 86 | Student Affairs Telegram channel, see below |
| `orientation_1446.pdf` | New Student Orientation Guide, Yanbu, 1446 AH | 13 | Student Affairs Telegram channel, see below  |
| `guidance_manual.pdf` | Student Guidance Manual, RCJY Yanbu | 19 | Student Affairs Telegram channel, see below  |
| `library_services_2024_2025.pdf` | Library Services Guide, 2024-2025 | 16 | Student Affairs Telegram channel, see below  |
| `academic_weeks_1448.pdf` | Academic calendar, 1448-1449 AH | 3 | Student Affairs Telegram channel, see below  |
| `organizational_regulations.pdf` | Organizational regulations for academic affairs | 58 | Regulations and guides page |
| `student_conduct_code.pdf` | Rules governing student conduct | 12 | Regulations and guides page |
| `student_charter.pdf` | Student charter | 8 | Regulations and guides page |
| `student_portal_guide.pdf` | Students' guide to the e-services portal, revision 5 | 42 | Regulations and guides page |

Filenames matter: the ground truth, the calibration script and `SHA256SUMS`
refer to them. After placing the files, check them from the repository root:

    sha256sum -c data/SHA256SUMS

Every line should end in `OK`. A mismatch means your copy differs from the one
measured here, and every published number may differ with it.

## Sources

The regulations and guides page is
<https://rcjy.gov.sa/ar/regulations-and-manuals/>. It publishes the
organizational regulations and the code of student conduct as separate files;
a 70-page copy circulating on WhatsApp binds the two together and is not used.

The student guide is not offered for download there. It was posted in the
Student Affairs announcements channel on Telegram,
[اعلانات وكالة شؤون الطلاب](https://t.me/SA_ED_YRC), but that post has since
been removed, and the guide is not on the college's site either (both checked
on 27 September 2026; the reason is unknown). The copy used here was saved
while it was available, and its original filename suggests 21 August 2025.
Its metadata names PDFium, the PDF engine inside Chrome and many other viewers,
as creator and producer, so it was probably saved through a viewer before
posting. `SHA256SUMS` identifies it exactly.

## Lexicon

The quality gate checks extracted words against the CAMeL Lab's Modern
Standard Arabic frequency list (Khalifa et al., 2021), licensed CC BY-SA 4.0.
It is downloaded, never committed:

    python3 scripts/fetch_lexicon.py

This saves a 69 MB zip to `data/external/`, verifies it against a pinned
checksum, and checks that the whole list is sorted by frequency.

## Tables typed by hand

One table's text layer lists its amounts apart from their rows: the fee table
on page 28 of the student guide. [`manual/tables.json`](manual/tables.json) holds it as typed in by hand
from the rendered page, with the lines of the text layer it stands for.
`daleel chunk` puts it in their place and marks its chunk
`manually_verified`, and stops if the page no longer holds those lines, since
the table would then describe a different page. Like the gold set's quotes,
it holds a few lines of the document, not the document.
