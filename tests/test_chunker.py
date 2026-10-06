"""Tests for chunking: lines, headings, units, merging, splitting and metadata.

Records are built by hand in the shape daleel extract writes, so every rule is
checked on a line or two of the corpus's own kind of text, without the corpus.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from daleel.chunk.chunker import (
    CLAUSE,
    FRAGMENT_WORDS,
    MAX_WORDS,
    MIN_CHUNK_WORDS,
    MIN_WORDS,
    OVERLAP,
    PROSE,
    TABLE,
    PageReader,
    Unit,
    article_label,
    corpus_chunks,
    document_chunks,
    document_units,
    merge,
    numbered_part,
    pieces,
    read_chunks,
    unit_line,
    windows,
    words,
    write_chunks,
)
from daleel.chunk.tables import ManualTable
from daleel.corpus import CALENDAR_DOCUMENT, COMMISSION, DOCUMENTS, YANBU
from daleel.eval.gold import CORPUS
from daleel.ingest.calendar import CalendarPage, CalendarRow

REGS = "organizational_regulations"
TATWEEL = chr(0x0640)


def layer(page: int, text: str, doc: str = REGS, score: float | None = 0.98765) -> dict:
    """A page whose text layer was kept."""
    gate = {"verdict": "trusted", "token_validity": score}
    return {"doc_id": doc, "page": page, "method": "text_layer", "text": text, "gate": gate}


def ocr(page: int, blocks: list[tuple[str, ...]], doc: str = "guidance_manual") -> dict:
    """A page read by OCR, as its layout's blocks: a category, a text and its HTML each."""
    layout = [
        {
            "category": block[0],
            "text": block[1],
            "box": None,
            "html": block[2] if len(block) > 2 else None,
            "engine_order": place,
        }
        for place, block in enumerate(blocks, start=1)
    ]
    return {
        "doc_id": doc,
        "page": page,
        "method": "ocr",
        "text": "\n\n".join(block[1] for block in blocks),
        "gate": {"verdict": "untrusted", "token_validity": None},
        "ocr": {"blocks": layout},
    }


def sentence(count: int, word: str = "كلمة") -> str:
    return " ".join([word] * count)


# --- article labels ------------------------------------------------------------


def test_a_label_at_the_end_of_a_line_heads_the_text_before_it() -> None:
    # The regulations' text layer reads an article's label, printed in the
    # margin, at the end of the article's first line.
    label = article_label("يتاح قبول الطالب وفق الضوابط المادة الحادية والعشرون:")
    assert label is not None
    assert label.label == "المادة الحادية والعشرون:"
    assert label.number == 21
    assert label.rest == "يتاح قبول الطالب وفق الضوابط"
    assert label.certain


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("المادة الأولى", 1),
        ("المادة العاشرة:", 10),
        ("المادة الحادية عشرة", 11),
        ("المادة الثانية والعشرون:", 22),
        ("المادة السادس والأربعون:", 46),
        ("المادة الستون:", 60),
        (f"الم{TATWEEL * 3}ادة الثالثة", 3),
    ],
)
def test_an_article_s_number_is_read_from_its_ordinal_words(text: str, number: int) -> None:
    label = article_label(text)
    assert label is not None
    assert label.number == number


def test_a_label_that_starts_a_line_with_a_colon_heads_the_rest() -> None:
    label = article_label("المادة الأولى: حقوق الطالب")
    assert label is not None
    assert (label.label, label.rest) == ("المادة الأولى:", "حقوق الطالب")


@pytest.mark.parametrize(
    "text",
    [
        "وفق ما ورد في المادة الحادية والثلاثين.",
        "مع مراعاة ما ورد في المادة الخامسة",
        "قيمة ما أتلفه وفق المادة (العاشرة)",
        "المادة الخامسة من هذه اللائحة",
        "تطبق أحكام المادة على الطلبة",
    ],
)
def test_a_reference_to_an_article_is_not_its_label(text: str) -> None:
    assert article_label(text) is None


def test_an_uncertain_label_must_follow_the_last_article() -> None:
    # The student guide names articles of the regulations in passing.
    reader = PageReader()
    assert not any(line.heading for line in reader.line("لما نصت عليه المادة السادسة"))
    reader.article = 5
    lines = reader.line("لما نصت عليه المادة السادسة")
    assert [line.text for line in lines if line.heading] == ["المادة السادسة"]
    assert reader.article == 6


def test_a_certain_label_restarts_the_count() -> None:
    reader = PageReader()
    reader.article = 72
    assert reader.line("المادة الثالثة:")[0].heading
    assert reader.article == 3


def test_a_numbered_part_ending_a_line_is_a_heading() -> None:
    found = numbered_part("تتم المعادلة وفقا للضوابط الآتية ثانيًا: شروط معادلة المقررات")
    assert found == ("تتم المعادلة وفقا للضوابط الآتية", "ثانيًا: شروط معادلة المقررات")
    assert numbered_part("أولاً: " + sentence(10)) is None


# --- unit lines ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "number"),
    [
        (".2 تشمل الرسوم الدراسية", "2"),
        (".21 الحصول على الحوافز", "21"),
        ("-1 تستوفى الرسوم", "1"),
        ("1) يعد الطالب المسجل", "1"),
        ("1- ضمان جودة العملية التعليمية", "1"),
        ("3 . ألا تتجاوز الوحدات", "3"),
        ("12. يكون الحد الأدنى", "12"),
    ],
)
def test_a_numbered_clause_starts_a_unit_with_its_number(text: str, number: str) -> None:
    line = unit_line(text)
    assert (line.starts, line.clause, line.clause_no) == (True, True, number)


@pytest.mark.parametrize(
    "text",
    [
        "3.75 من 4 في آخر فصلين",
        "2025 م",
        "-أو من يفوضه.-",
        "(15) يوما ولا تحتسب",
        "أ. إيقاف الطالب عن التسجيل",
        "ب- أن تكون لغة الدراسة الإنجليزية",
        "12 وحدة دراسية تخصصية",
    ],
)
def test_other_lines_continue_the_unit(text: str) -> None:
    line = unit_line(text)
    assert (line.starts, line.clause_no) == (False, None)


def test_a_bullet_starts_a_clause_without_a_number() -> None:
    line = unit_line("• الكلية/المعهد: كليات ومعاهد الهيئة الملكية")
    assert (line.starts, line.clause, line.clause_no) == (True, True, None)


# --- units ---------------------------------------------------------------------


def test_articles_and_clauses_become_units_under_their_headings() -> None:
    text = "\n".join(
        [
            "09",
            "يتاح قبول الطالب في البرنامج وفق الضوابط المادة الحادية والعشرون:",
            "والآليات التي يقرها المجلس.",
            "1) يعد الطالب المسجل طالبا منتظما المادة الثانية والعشرون:",
            "بجميع الحقوق.",
            "2) يعد الطالب غير المسجل مقيدا في الحالات الآتية:",
            "أ. إيقاف الطالب.",
            "ب. اعتذار الطالب.",
        ]
    )
    units = document_units([layer(12, text)])
    assert [(unit.heading, unit.clause_no, unit.text) for unit in units] == [
        (
            "المادة الحادية والعشرون",
            None,
            "يتاح قبول الطالب في البرنامج وفق الضوابط والآليات التي يقرها المجلس.",
        ),
        ("المادة الثانية والعشرون", "1", "1) يعد الطالب المسجل طالبا منتظما بجميع الحقوق."),
        (
            "المادة الثانية والعشرون",
            "2",
            "2) يعد الطالب غير المسجل مقيدا في الحالات الآتية: أ. إيقاف الطالب. ب. اعتذار الطالب.",
        ),
    ]
    assert {unit.content_type for unit in units} == {CLAUSE}


def test_a_heading_carries_over_to_the_next_page() -> None:
    pages = [layer(1, "المادة الأولى\nيقصد بالألفاظ الآتية"), layer(2, "المعاني المبينة أمامها")]
    assert [(unit.page, unit.heading) for unit in document_units(pages)] == [
        (1, "المادة الأولى"),
        (2, "المادة الأولى"),
    ]


def test_a_line_of_numbers_ends_a_unit_and_is_left_out() -> None:
    # The student guide prints each item's number beside it, after it in its
    # text layer.
    units = document_units([layer(10, "شرط أول للقبول\n01\nشرط ثان للقبول\n3 03\n750")])
    assert [unit.text for unit in units] == ["شرط أول للقبول", "شرط ثان للقبول", "750"]
    assert [unit.clause_no for unit in units] == [None, None, None]
    assert {unit.content_type for unit in units} == {PROSE}


def test_lines_with_no_letter_or_digit_are_left_out() -> None:
    units = document_units([layer(3, "نص أول\n.\n. .\nنص ثان")])
    assert [unit.text for unit in units] == ["نص أول نص ثان"]


def test_running_headers_are_left_out_where_there_are_pages_enough_to_tell() -> None:
    header = "\n".join(["دليل استخدام", "بوابة الخدمات", "2025", ""])
    pages = [
        layer(page, header + f"خطوة رقم {page}", "student_portal_guide") for page in range(1, 7)
    ]
    assert [unit.text for unit in document_units(pages)] == [
        f"خطوة رقم {page}" for page in range(1, 7)
    ]
    few = document_units(pages[:5])
    assert few[0].text == "دليل استخدام بوابة الخدمات 2025 خطوة رقم 1"


def test_a_table_of_contents_names_the_sections_and_is_not_chunked() -> None:
    doc = "student_guide_2025"
    assert DOCUMENTS[doc].contents_page == 3
    pages = [
        layer(1, "دليل الطالب", doc),
        layer(2, "صناعة وحياة", doc),
        layer(3, "\n".join(["الفهرس", "المواظبة", "التقديرات", "26", "27", ".", "."]), doc),
        layer(
            4,
            "\n".join(
                [
                    "المواظبة",
                    ".1 يحرم الطالب عند تجاوز الغياب",
                    "التقديرات",
                    ".1 يمنح الطالب تقديرا",
                ]
            ),
            doc,
        ),
        layer(5, "الفهرس يذكر هنا عرضا", doc),
    ]
    units = document_units(pages)
    assert [(unit.page, unit.heading, unit.clause_no) for unit in units] == [
        (1, None, None),
        (2, None, None),
        (4, "المواظبة", "1"),
        (4, "التقديرات", "1"),
        (5, "التقديرات", None),
    ]


def test_a_document_read_without_its_table_of_contents_has_no_sections() -> None:
    units = document_units([layer(4, "المواظبة", "student_guide_2025")])
    assert [(unit.heading, unit.text) for unit in units] == [(None, "المواظبة")]


def test_a_layout_gives_headings_list_items_and_tables() -> None:
    page = ocr(
        6,
        [
            ("Page-header", "الهيئة الملكية\nللجبيل وينبع"),
            ("Section-header", "الحرمان (DN):"),
            ("Picture", ""),
            ("List-item", "يحرم الطالب من دخول الاختبار النهائي."),
            ("List-item", "إذا زادت نسبة غياب الطالب بعذر عن 25%."),
            ("Formula", "$$ \\text{x} $$"),
            ("Table", "\n".join(["الفئة عدد الكتب", "الطلبة 5 كتب"])),
            ("Text", "سطر أول\nسطر ثان"),
            ("Page-footer", "6"),
        ],
    )
    units = document_units([page])
    assert [(unit.heading, unit.content_type, unit.text) for unit in units] == [
        ("الحرمان (DN)", CLAUSE, "يحرم الطالب من دخول الاختبار النهائي."),
        ("الحرمان (DN)", CLAUSE, "إذا زادت نسبة غياب الطالب بعذر عن 25%."),
        ("الحرمان (DN)", TABLE, "\n".join(["الفئة عدد الكتب", "الطلبة 5 كتب"])),
        ("الحرمان (DN)", PROSE, "سطر أول سطر ثان"),
    ]


def test_a_table_in_the_layout_becomes_a_sentence_per_row() -> None:
    table = (
        "<table><thead><tr><td>الفئة</td><td>عدد الكتب</td></tr></thead>"
        "<tbody><tr><td>الطلبة</td><td>5 كتب</td></tr><tr><td>الموظفون</td><td>3 كتب</td></tr>"
        "</tbody></table>"
    )
    page = ocr(8, [("Section-header", "مدة الاعارة"), ("Table", "الفئة عدد الكتب", table)])
    (unit,) = document_units([page])
    assert (unit.heading, unit.content_type) == ("مدة الاعارة", TABLE)
    assert unit.lines == ["الفئة: الطلبة، عدد الكتب: 5 كتب", "الفئة: الموظفون، عدد الكتب: 3 كتب"]


def test_a_block_called_a_heading_but_long_is_text() -> None:
    units = document_units([ocr(2, [("Section-header", sentence(20))])])
    assert [unit.text for unit in units] == [sentence(20)]


def test_an_article_heading_in_a_layout_counts_for_the_labels_after_it() -> None:
    pages = [
        ocr(13, [("Section-header", "المادة الخامسة والعشرون:"), ("Text", "يلتزم الطالب")], REGS),
        layer(14, "تشكل لجنة بمسمى لجنة الحالات المادة السادس والعشرون"),
    ]
    assert [unit.heading for unit in document_units(pages)] == [
        "المادة الخامسة والعشرون",
        "المادة السادس والعشرون",
    ]


# --- merging and splitting -----------------------------------------------------


def unit(text: str, page: int = 1, heading: str | None = "h", kind: str = CLAUSE) -> Unit:
    return Unit(page, heading, kind, [text])


def test_short_units_merge_under_one_heading_on_one_page() -> None:
    short = sentence(MIN_WORDS // 2)
    drafts = merge([unit(short), unit(short), unit(short), unit(short)])
    assert [len(draft.texts) for draft in drafts] == [2, 2]
    shorter = sentence(MIN_WORDS // 3)
    drafts = merge([unit(shorter), unit(shorter), unit(shorter), unit(shorter)])
    assert [len(draft.texts) for draft in drafts] == [4]


def test_units_do_not_merge_across_headings_pages_or_tables() -> None:
    short = sentence(5)
    units = [
        unit(short),
        unit(short, heading="other"),
        unit(short, heading="other", page=2),
        unit(short, heading="other", page=2, kind=TABLE),
        unit(short, heading="other", page=2),
    ]
    assert len(merge(units)) == 5


def test_a_fragment_joins_the_unit_before_it() -> None:
    drafts = merge([unit(sentence(MIN_WORDS + 5)), unit(sentence(FRAGMENT_WORDS - 1))])
    assert len(drafts) == 1
    drafts = merge([unit(sentence(MIN_WORDS + 5)), unit(sentence(FRAGMENT_WORDS))])
    assert len(drafts) == 2


def test_no_merge_makes_a_chunk_longer_than_the_most() -> None:
    drafts = merge([unit(sentence(MIN_WORDS - 1)), unit(sentence(MAX_WORDS - MIN_WORDS + 2))])
    assert len(drafts) == 2


def test_a_merged_chunk_is_a_clause_if_any_part_is_and_spans_their_numbers() -> None:
    parts = [unit("مقدمة قصيرة", kind=PROSE), unit(".1 أول"), unit(".2 ثان"), unit(".3 ثالث")]
    for part, number in zip(parts, [None, "1", "2", "3"], strict=True):
        part.clause_no = number
    (draft,) = merge(parts)
    assert (draft.content_type, draft.clause_no) == (CLAUSE, "1-3")
    assert draft.texts[0] == "مقدمة قصيرة"


def test_a_long_unit_is_cut_into_overlapping_pieces() -> None:
    tokens = [f"w{number}" for number in range(400)]
    long = unit(" ".join(tokens))
    parts = pieces(long)
    assert all(words(part.text) <= MAX_WORDS for part in parts)
    texts = [part.text.split() for part in parts]
    assert texts[0][0] == "w0" and texts[-1][-1] == "w399"
    for before, after in pairwise(texts):
        assert before[-OVERLAP:] == after[:OVERLAP]


def test_a_piece_ends_at_a_sentence_where_one_falls_late_enough() -> None:
    ends = [False] * 400
    ends[149] = True
    assert windows(400, ends)[0] == (0, 150)
    ends = [False] * 400
    ends[50] = True
    assert windows(400, ends)[0] == (0, MAX_WORDS)


def test_a_long_table_is_cut_between_its_rows() -> None:
    rows = [sentence(70) for _ in range(5)]
    table = Unit(1, "h", TABLE, rows, manual=True)
    parts = pieces(table)
    assert [part.lines for part in parts] == [rows[:2], rows[2:4], rows[4:]]
    assert all(part.manual and part.content_type == TABLE for part in parts)
    short = Unit(1, "h", TABLE, rows[:2])
    assert pieces(short) == [short]


# --- tables typed by hand and the calendar --------------------------------------

GUIDE = "student_guide_2025"


def fee_table(*replaces: str) -> ManualTable:
    return ManualTable(
        GUIDE,
        28,
        "تستوفى الرسوم حسب الآتي:",
        ("الفئة", "المبلغ"),
        (("الطالب السعودي", "250"), ("مقيم", "1100")),
        replaces,
        "by hand",
    )


def test_a_table_typed_by_hand_takes_the_place_of_its_scattered_lines() -> None:
    text = "\n".join(
        [
            ".4 يجب سداد الرسوم كاملة",
            "-1 تستوفى الرسوم حسب الآتي:",
            "الطالب السعودي",
            "250",
            "المواظبة",
            "مقيم",
            "1100",
            "وتبقى هذه الفقرة بعد الجدول كما هي",
        ]
    )
    lines = ["-1 تستوفى الرسوم حسب الآتي:", "الطالب السعودي", "250", "المواظبة", "مقيم", "1100"]
    pages = [layer(3, "\n".join(["الفهرس", "المواظبة"]), GUIDE), layer(28, text, GUIDE)]
    chunks = document_chunks(pages, [fee_table(*lines)])
    assert [(chunk["content_type"], chunk["manually_verified"]) for chunk in chunks] == [
        (CLAUSE, False),
        (TABLE, True),
        (PROSE, False),
    ]
    assert chunks[1]["text"] == "\n".join(
        [
            "تستوفى الرسوم حسب الآتي، الفئة: الطالب السعودي، المبلغ: 250",
            "تستوفى الرسوم حسب الآتي، الفئة: مقيم، المبلغ: 1100",
        ]
    )
    # A scattered cell that is also one of the guide's sections, as برنامج
    # الدراسة بمقابل مالي is, starts no section: the table holds it.
    assert {chunk["section_heading"] for chunk in chunks} == {None}


def test_a_table_typed_by_hand_must_still_match_its_page() -> None:
    page = layer(28, "\n".join(["الطالب السعودي", "250"]), GUIDE)
    with pytest.raises(ValueError, match="page 28 no longer holds the lines"):
        document_chunks([page], [fee_table("الطالب السعودي", "250", "مقيم")])


def test_tables_typed_by_hand_apply_to_their_own_document_and_page() -> None:
    table = fee_table("250")
    assert [chunk["text"] for chunk in document_chunks([layer(28, "250 ريال لكل وحدة")], [table])]
    page = layer(27, "250 ريال لكل وحدة", GUIDE)
    assert document_chunks([page], [table])[0]["manually_verified"] is False


def calendar_page() -> CalendarPage:
    meem, heh = chr(0x0645), chr(0x0647)
    start = CalendarRow(
        "بداية الفصل الدراسي الأول",
        "Start of First Semester",
        "الأحد",
        "Sun",
        "2026/08/23" + meem,
        "1448/03/10" + heh,
    )
    drop = CalendarRow(
        "نهاية فترة حذف وإضافة المقررات",
        "Last day for add / drop courses",
        "الخميس",
        "Thu",
        "2026/08/27" + meem,
        "1448/03/14" + heh,
    )
    return CalendarPage(1, "الفصل الدراسي الأول (481)", "First Semester (481)", (start, drop))


def test_each_calendar_card_is_a_chunk_under_its_semester() -> None:
    record = layer(1, "نص الصفحة كما استخرج", CALENDAR_DOCUMENT, score=1.0)
    chunks = document_chunks([record], calendar=[calendar_page()])
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "academic_weeks_1448_p1_c1",
        "academic_weeks_1448_p1_c2",
    ]
    assert {chunk["section_heading"] for chunk in chunks} == {"الفصل الدراسي الأول (481)"}
    assert {chunk["content_type"] for chunk in chunks} == {TABLE}
    assert "نهاية فترة حذف وإضافة المقررات" in chunks[1]["text"]
    assert "2026/08/27" in chunks[1]["text"]
    assert chunks[1]["extraction_method"] == "text_layer"


def test_the_cards_go_to_the_calendar_alone(tmp_path: Path) -> None:
    folder = tmp_path / "records"
    folder.mkdir()
    for doc in (CALENDAR_DOCUMENT, "student_charter"):
        record = layer(1, "نص من صفحة الوثيقة الأولى", doc)
        (folder / f"{doc}.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    chunks = corpus_chunks(folder, calendar=[calendar_page()])
    assert [chunk["doc_id"] for chunk in chunks] == [CALENDAR_DOCUMENT] * 2 + ["student_charter"]


# --- chunks --------------------------------------------------------------------


def test_chunks_carry_what_a_citation_needs() -> None:
    text = "المادة الثانية والعشرون:\n.1 " + sentence(50) + "\n.2 " + sentence(50)
    chunks = document_chunks([layer(12, text)])
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "organizational_regulations_p12_c1",
        "organizational_regulations_p12_c2",
    ]
    first = chunks[0]
    assert first == {
        "chunk_id": "organizational_regulations_p12_c1",
        "doc_id": REGS,
        "doc_title": DOCUMENTS[REGS].title,
        "scope": COMMISSION,
        "effective": DOCUMENTS[REGS].effective,
        "page": 12,
        "section_heading": "المادة الثانية والعشرون",
        "clause_no": "1",
        "content_type": CLAUSE,
        "extraction_method": "text_layer",
        "gate_score": 0.988,
        "manually_verified": False,
        "text": ".1 " + sentence(50),
    }


def test_a_page_read_by_ocr_keeps_its_method_and_has_no_score() -> None:
    (chunk,) = document_chunks([ocr(5, [("Text", sentence(10))])])
    assert (chunk["extraction_method"], chunk["gate_score"]) == ("ocr", None)
    assert chunk["scope"] == YANBU


def test_chunks_too_short_to_hold_anything_are_dropped() -> None:
    pages = [layer(1, "2025"), layer(2, sentence(MIN_CHUNK_WORDS))]
    assert [chunk["page"] for chunk in document_chunks(pages)] == [2]


def test_a_document_outside_the_corpus_is_refused() -> None:
    with pytest.raises(ValueError, match="not a document of the corpus"):
        document_chunks([layer(1, "نص", "unknown")])
    assert document_chunks([]) == []


def test_chunks_are_written_and_read_back(tmp_path: Path) -> None:
    folder = tmp_path / "records"
    folder.mkdir()
    for doc, page in [(REGS, 2), (REGS, 1), ("student_charter", 1)]:
        with (folder / f"{doc}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(layer(page, sentence(5), doc), ensure_ascii=False) + "\n")
    chunks = corpus_chunks(folder)
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "organizational_regulations_p1_c1",
        "organizational_regulations_p2_c1",
        "student_charter_p1_c1",
    ]
    path = write_chunks(chunks, tmp_path / "processed" / "chunks.jsonl")
    assert read_chunks(path) == chunks
    assert not list(path.parent.glob("*.tmp"))


# --- the corpus ----------------------------------------------------------------


def test_every_document_of_the_corpus_is_described() -> None:
    assert DOCUMENTS.keys() == CORPUS.keys()
    for doc_id, document in DOCUMENTS.items():
        assert document.title
        assert document.scope in (COMMISSION, YANBU)
        if document.contents_page is not None:
            assert 1 <= document.contents_page <= CORPUS[doc_id]
