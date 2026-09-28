"""Tests for the Tesseract wrapper.

Most tests stand in for the program, so they run where Tesseract is not
installed, CI included. One runs the real engine when it is installed with the
English model: an English line rendered from a test PDF proves the plumbing,
while Arabic accuracy is what the OCR evaluation measures.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daleel.ocr import tesseract
from daleel.ocr.render import render_page
from daleel.ocr.tesseract import (
    DEFAULT,
    Settings,
    TesseractError,
    model_hashes,
    parse_tessdata,
    parse_version,
    recognize,
)

LIST_LANGS = 'List of available languages in "/usr/share/tesseract-ocr/5/tessdata/" (3):\n'


def fake_run(
    stdout: bytes = b"", stderr: bytes = b"", calls: list[Any] | None = None
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A stand-in for subprocess.run that records each call and succeeds."""

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append((command, kwargs.get("input")))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)

    return run


def test_the_command_reads_stdin_and_states_the_resolution() -> None:
    assert Settings(lang="ara", psm=6, dpi=300).command() == [
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        "ara",
        "--psm",
        "6",
        "--dpi",
        "300",
    ]


def test_the_default_is_arabic_with_automatic_layout_at_300_dpi() -> None:
    assert Settings(lang="ara", psm=3, dpi=300) == DEFAULT


def test_the_image_goes_in_and_the_text_comes_out(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        tesseract.subprocess, "run", fake_run(stdout="درجة\n".encode(), calls=calls)
    )
    result = recognize(b"png bytes")
    assert calls == [(DEFAULT.command(), b"png bytes")]
    assert result.text == "درجة\n"
    assert result.seconds >= 0
    assert result.warnings == ""


def test_warnings_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tesseract.subprocess, "run", fake_run(stderr=b"Estimating resolution as 462\n")
    )
    assert recognize(b"").warnings == "Estimating resolution as 462"


def test_a_failure_carries_tesseract_s_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command, stderr=b"Failed loading language 'xyz'")

    monkeypatch.setattr(tesseract.subprocess, "run", failing)
    with pytest.raises(TesseractError, match="Failed loading language"):
        recognize(b"", Settings(lang="xyz"))


def test_a_missing_program_is_reported() -> None:
    with pytest.raises(TesseractError, match="not installed"):
        recognize(b"", binary="no-such-tesseract")


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("tesseract 5.3.4\n leptonica-1.82.0\n  libgif 5.2.1 : libjpeg 8d\n", "5.3.4"),
        ("tesseract 5.5.1\n leptonica-1.85.0\n", "5.5.1"),
    ],
)
def test_the_version_is_the_first_line(output: str, expected: str) -> None:
    assert parse_version(output) == expected


@pytest.mark.parametrize("output", ["", "leptonica-1.82.0\n", "tesseract\n"])
def test_unexpected_version_output_is_an_error(output: str) -> None:
    with pytest.raises(TesseractError, match="unexpected version"):
        parse_version(output)


def test_the_model_directory_comes_from_the_language_list() -> None:
    listing = LIST_LANGS + "ara\neng\nosd\n"
    assert parse_tessdata(listing) == Path("/usr/share/tesseract-ocr/5/tessdata/")


def test_a_language_list_without_a_directory_is_an_error() -> None:
    with pytest.raises(TesseractError, match="no model directory"):
        parse_tessdata("ara\neng\n")


def test_every_model_a_setting_loads_is_hashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "ara.traineddata").write_bytes(b"arabic weights")
    (tmp_path / "eng.traineddata").write_bytes(b"english weights")
    listing = f'List of available languages in "{tmp_path}/" (2):\nara\neng\n'
    monkeypatch.setattr(tesseract.subprocess, "run", fake_run(stdout=listing.encode()))
    assert model_hashes("ara") == {"ara": hashlib.sha256(b"arabic weights").hexdigest()}
    assert list(model_hashes("ara+eng")) == ["ara", "eng"]


def english_tesseract() -> bool:
    if shutil.which("tesseract") is None:
        return False
    listing = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    return "eng" in (listing.stdout + listing.stderr).split()


@pytest.mark.skipif(not english_tesseract(), reason="needs Tesseract with the English model")
def test_real_tesseract_reads_a_rendered_line(make_pdf: Callable[..., Path]) -> None:
    png = render_page(make_pdf(["DALEEL READS THIS LINE"]), 1).png()
    result = recognize(png, Settings(lang="eng", psm=6))
    assert "DALEEL READS THIS LINE" in result.text
    # The image states its resolution and so does the command line, so
    # Tesseract has nothing to estimate or reject.
    assert "resolution" not in result.warnings.lower()
