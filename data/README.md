# Source documents

The PDFs this project reads are internal RCJY college publications. They are
**not committed to this repository** and `data/raw/` is gitignored.

To run the pipeline, place the following in `data/raw/`:

| Expected filename | Document | Pages |
|---|---|---|
| `student_guide_2025.pdf` | Student Guide, RCJY colleges and institutes, 2025 | 86 |
| `orientation_1446.pdf` | New Student Orientation Guide, Yanbu, 1446 AH | 13 |
| `guidance_manual.pdf` | Student Guidance Manual, RCJY Yanbu | 19 |
| `library_services_2024_2025.pdf` | Library Services Guide, 2024-2025 | 16 |
| `academic_weeks_1448.pdf` | Academic week distribution, 1448 AH | 3 |

All five are obtainable from the college's student affairs office or the RCJY
education portal.

Filenames matter: the ingestion code keys document metadata off them.
