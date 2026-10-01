"""Tests for the calendar CSV checks.

Date cells are copied from academic_weeks_1448_p01, so these tests encode what
the page prints. Note that printed ranges run end first: no ordering is
imposed on ground truth (rule 6.9).

The era suffixes appear as \u0645 (meem) and \u0647 (heh) rather than as
literal Arabic, so each cell reads unambiguously in a file that is otherwise
ASCII, and so no editor reorders these strings for display.
"""

from __future__ import annotations

import datetime as dt

import pytest

from daleel.eval.calendar_checks import (
    DateCell,
    ics_spans,
    read_date_cell,
    span_days,
    span_disagreement,
    weekday_disagreement,
)


def test_single_date_has_one_end() -> None:
    cell = read_date_cell("2026/08/23\u0645")
    assert cell.ends == ((2026, 8, 23),)
    assert cell.problem is None
    assert not cell.is_range


def test_range_printed_end_first_is_accepted_as_printed() -> None:
    # The National Day holiday, printed 26-23: the later day comes first and
    # that is what ground truth records.
    cell = read_date_cell("2026/09/26-23\u0645")
    assert cell.ends == ((2026, 9, 26), (2026, 9, 23))
    assert cell.problem is None


def test_range_crossing_months_carries_both_months() -> None:
    # The mid-year holiday's Hijri range, which crosses from one month to the next.
    assert read_date_cell("1448/08/08-07/30\u0647").ends == ((1448, 8, 8), (1448, 7, 30))


def test_a_space_before_the_suffix_is_read() -> None:
    # Page 2 prints the start of the second semester with one.
    assert read_date_cell("2027/01/17 \u0645").ends == ((2027, 1, 17),)


def test_months_and_days_of_one_digit_are_read() -> None:
    # Page 2 prints the last day for grades as 1449/1/10, and Eid al-Fitr as a
    # range back into the second month.
    assert read_date_cell("1449/1/10\u0647").ends == ((1449, 1, 10),)
    assert read_date_cell("2027/03/13-2/26\u0645").ends == ((2027, 3, 13), (2027, 2, 26))


def test_a_hijri_range_over_two_lines_is_read() -> None:
    # Page 2's final exams: the range crosses into 1449, a full date a line.
    cell = read_date_cell("1448/12/26\u0647 1449/01/09-\u0647")
    assert cell.ends == ((1448, 12, 26), (1449, 1, 9))
    assert span_days(cell, hijri=True) == 13
    assert span_disagreement(read_date_cell("2027/06/14-01\u0645"), cell) is None


def test_surrounding_whitespace_is_ignored() -> None:
    assert read_date_cell("  2026/08/23م  ").ends == ((2026, 8, 23),)


@pytest.mark.parametrize(
    "cell",
    [
        "2026/08/23",  # no era suffix
        "2026/123/23\u0645",  # three-digit month
        "23-2026/09/26\u0645",  # the start day moved to the front
        "2026-08-23\u0645",  # wrong separator
        "1448/12/26\u0647 1449/01/09-\u0645",  # a range over two lines, its suffixes differing
        "",
    ],
    ids=[
        "no_suffix",
        "three_digit_month",
        "start_day_first",
        "hyphen_separators",
        "two_suffixes",
        "empty",
    ],
)
def test_malformed_cells_are_reported(cell: str) -> None:
    result = read_date_cell(cell)
    assert result.problem is not None
    assert "does not match" in result.problem


def test_impossible_month_is_reported() -> None:
    result = read_date_cell("2026/13/23\u0645")
    assert result.problem is not None
    assert "out of range" in result.problem


def test_single_date_spans_no_days() -> None:
    assert span_days(read_date_cell("2026/08/23\u0645"), hijri=False) == 0


def test_gregorian_span_uses_real_dates() -> None:
    # 20 to 28 November, whichever end was printed first.
    assert span_days(read_date_cell("2026/11/28-20\u0645"), hijri=False) == 8


def test_hijri_span_within_one_month() -> None:
    assert span_days(read_date_cell("1448/06/18-10\u0647"), hijri=True) == 8


def test_hijri_span_across_months_assumes_thirty_day_months() -> None:
    # 30 Rajab to 8 Sha'ban, which the Gregorian range gives as 8 days.
    assert span_days(read_date_cell("1448/08/08-07/30\u0647"), hijri=True) == 8


def test_matching_calendars_do_not_disagree() -> None:
    assert (
        span_disagreement(
            read_date_cell("2027/01/16-08\u0645"), read_date_cell("1448/08/08-07/30\u0647")
        )
        is None
    )


def test_scrambled_hijri_cell_is_caught() -> None:
    # The real error found in the transcription: both pairs landed in the same
    # Hijri month, which makes the range far too long.
    problem = span_disagreement(
        read_date_cell("2027/01/16-08\u0645"), read_date_cell("1448/08/30-08/07\u0647")
    )
    assert problem is not None
    assert "23" in problem


def test_a_day_of_slack_is_allowed() -> None:
    # Hijri months are 29 or 30 days, so a one-day difference proves nothing.
    assert (
        span_disagreement(
            read_date_cell("2026/09/26-23\u0645"), read_date_cell("1448/04/15-12\u0647")
        )
        is None
    )


def test_malformed_cells_never_report_a_disagreement() -> None:
    assert (
        span_disagreement(read_date_cell("nonsense"), read_date_cell("1448/04/15-12\u0647")) is None
    )


def test_all_day_event_end_date_is_exclusive() -> None:
    raw = (
        b"BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\n"
        b"DTSTART;VALUE=DATE:20260923\r\nDTEND;VALUE=DATE:20260927\r\n"
        b"END:VEVENT\r\nEND:VCALENDAR\r\n"
    )
    days, spans = ics_spans(raw)
    # The file says the 27th; the holiday's last day is the 26th.
    assert days == {dt.date(2026, 9, 23), dt.date(2026, 9, 26)}
    assert spans == {frozenset({dt.date(2026, 9, 23), dt.date(2026, 9, 26)})}


def test_timed_event_end_is_not_shifted() -> None:
    raw = (
        b"BEGIN:VEVENT\r\n"
        b"DTSTART;TZID=Asia/Riyadh:20270103T120000\r\n"
        b"DTEND;TZID=Asia/Riyadh:20270103T130000\r\n"
        b"END:VEVENT\r\n"
    )
    days, _ = ics_spans(raw)
    assert days == {dt.date(2027, 1, 3)}


def test_folded_lines_do_not_break_parsing() -> None:
    # A fold is a CRLF followed by a space, and it may split a multi-byte
    # character, which is why unfolding happens on bytes.
    raw = (
        b"BEGIN:VEVENT\r\nSUMMARY:\xd8\xa5\xd8\xac\xd8\xa7\r\n \xd8\xb2\xd8\xa9\r\n"
        b"DTSTART;VALUE=DATE:20260823\r\nDTEND;VALUE=DATE:20260824\r\n"
        b"END:VEVENT\r\n"
    )
    days, _ = ics_spans(raw)
    assert days == {dt.date(2026, 8, 23)}


def test_empty_calendar_has_nothing_in_it() -> None:
    assert ics_spans(b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n") == (set(), set())


def test_date_cell_defaults_are_empty() -> None:
    assert DateCell().ends == ()
    assert DateCell().problem is None


def test_days_that_fall_on_their_dates_agree() -> None:
    assert weekday_disagreement("Sun", read_date_cell("2026/08/23\u0645")) is None


def test_a_range_s_days_run_first_day_first_whichever_end_its_dates_print_first() -> None:
    # The National Day holiday: printed 26-23, Wednesday the 23rd to Saturday.
    assert weekday_disagreement("Wed - Sat", read_date_cell("2026/09/26-23\u0645")) is None


def test_days_from_another_card_are_caught() -> None:
    problem = weekday_disagreement("Thu", read_date_cell("2026/08/23\u0645"))
    assert problem is not None
    assert "Sun" in problem


def test_a_range_s_days_in_the_wrong_order_are_caught() -> None:
    assert weekday_disagreement("Sat - Wed", read_date_cell("2026/09/26-23\u0645")) is not None


def test_dates_that_do_not_exist_are_reported() -> None:
    problem = weekday_disagreement("Sun", read_date_cell("2027/02/30\u0645"))
    assert problem == "the dates do not exist"


def test_unreadable_dates_report_no_weekday() -> None:
    assert weekday_disagreement("Sun", read_date_cell("nonsense")) is None
