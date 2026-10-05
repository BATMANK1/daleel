"""The nine documents of the corpus: what each is, whom it governs, and its date.

Every chunk carries its document's title, scope and date, so that an answer
can say where a rule comes from, and so that two documents that disagree can
be weighed. The organizational regulations govern every college and institute
of the Royal Commission; the orientation guide speaks for Yanbu alone.

Titles are as printed on each cover. Dates are as printed too, in whichever
calendar the document prints them, and None where it prints none. The academic
calendar names no city and carries only the Commission's name, so it is taken
as the Commission's.
"""

from __future__ import annotations

from dataclasses import dataclass

COMMISSION = "commission"
YANBU = "yanbu"


@dataclass(frozen=True)
class Document:
    title: str
    # Whom it governs: every college and institute of the Royal Commission for
    # Jubail and Yanbu, or those in Yanbu.
    scope: str
    effective: str | None
    # The PDF page of the document's own table of contents, whose entries are
    # its section headings, where it has one that lists them as text.
    contents_page: int | None = None


DOCUMENTS = {
    "academic_weeks_1448": Document(
        "التقويم الأكاديمي للعام الدراسي 1448 - 1449 هـ", COMMISSION, "1448-1449 AH / 2026-2027"
    ),
    "guidance_manual": Document(
        "الدليل الإرشادي لطلبة كليات ومعاهد الهيئة الملكية بينبع", YANBU, None
    ),
    "library_services_2024_2025": Document(
        "دليل الطالب للاستفادة من خدمات المكتبات", YANBU, "2024-2025"
    ),
    "organizational_regulations": Document(
        "اللائحة التنظيمية للشؤون الدراسية والأكاديمية", COMMISSION, "1447 AH / 2025"
    ),
    "orientation_1446": Document("الدليل الإرشادي للطالب المستجد", YANBU, "1446 AH"),
    "student_charter": Document("ميثاق الطالب", COMMISSION, "2025"),
    "student_conduct_code": Document("القواعد المنظمة لسلوك الطلبة", COMMISSION, "2025"),
    "student_guide_2025": Document(
        "دليل الطالب في كليات ومعاهد الهيئة الملكية للجبيل وينبع",
        COMMISSION,
        "2025",
        contents_page=3,
    ),
    "student_portal_guide": Document(
        "دليل استخدام بوابة الخدمات الإلكترونية للطلبة", COMMISSION, "2025"
    ),
}
