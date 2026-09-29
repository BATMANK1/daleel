"""Tests for the PaddleOCR engine.

PaddleOCR is not installed for these tests, CI included: a stand-in answers
predict() the way PaddleOCR 3.7 does, with one result per page holding the
recognized lines and their polygons, left box first on a shared row.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from daleel.ocr import paddle
from daleel.ocr.paddle import (
    DEFAULT,
    PaddleEngine,
    Settings,
    folder_sha256,
    installed,
    model_folder,
)

LEFT_POLY = [[130, 1000], [900, 1000], [900, 1055], [130, 1055]]
RIGHT_POLY = [[1000, 1000], [2340, 1000], [2340, 1055], [1000, 1055]]


class FakeOCR:
    """Answers predict() like paddleocr.PaddleOCR, recording what it was given."""

    def __init__(self, texts: list[str], polys: list[Any]) -> None:
        self.result = {"rec_texts": texts, "rec_polys": polys}
        self.pages: list[bytes] = []

    def predict(self, page: str) -> list[dict[str, Any]]:
        self.pages.append(Path(page).read_bytes())
        return [self.result]


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A PaddleX cache holding both default models."""
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    for name in (DEFAULT.detection_model, DEFAULT.recognition_model):
        folder = tmp_path / "official_models" / name
        folder.mkdir(parents=True)
        (folder / "inference.pdiparams").write_bytes(name.encode())
    return tmp_path


def test_the_models_are_named_and_preprocessing_is_off() -> None:
    options = Settings().options()
    assert options["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert options["text_recognition_model_name"] == "arabic_PP-OCRv5_mobile_rec"
    assert not options["use_doc_orientation_classify"]
    assert not options["use_doc_unwarping"]
    assert not options["use_textline_orientation"]
    # The backend that turns each Arabic line into reading order.
    assert options["engine"] == "paddle_static"


def test_a_page_is_read_in_arabic_reading_order(models: Path) -> None:
    ocr = FakeOCR(["الدراسية.", "المادة الثانية والخمسون:"], [LEFT_POLY, RIGHT_POLY])
    result = PaddleEngine(ocr=ocr).recognize(b"png bytes")
    assert result.text == "المادة الثانية والخمسون: الدراسية."
    assert result.seconds >= 0


def test_the_page_reaches_paddleocr_as_the_png_given(models: Path) -> None:
    ocr = FakeOCR([], [])
    PaddleEngine(ocr=ocr).recognize(b"png bytes")
    assert ocr.pages == [b"png bytes"]


def test_a_page_with_no_text_reads_as_empty(models: Path) -> None:
    assert PaddleEngine(ocr=FakeOCR([], [])).recognize(b"png").text == ""


def test_building_the_engine_passes_the_settings_to_paddleocr(
    models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}

    class PaddleOCR:
        def __init__(self, **options: Any) -> None:
            received.update(options)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=PaddleOCR))
    PaddleEngine(Settings(device="gpu:0"))
    assert received == Settings(device="gpu:0").options()


def test_models_are_hashed_from_the_paddlex_cache(models: Path) -> None:
    engine = PaddleEngine(ocr=FakeOCR([], []))
    assert list(engine.models) == [DEFAULT.detection_model, DEFAULT.recognition_model]
    assert engine.models[DEFAULT.recognition_model] == folder_sha256(
        models / "official_models" / DEFAULT.recognition_model
    )
    assert engine.tag().endswith("-cpu")
    assert "arabic_PP-OCRv5_mobile_rec sha256" in engine.describe()


def test_a_missing_model_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no model in"):
        PaddleEngine(ocr=FakeOCR([], []))


def test_the_cache_defaults_to_the_home_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    assert model_folder("m") == Path.home() / ".paddlex" / "official_models" / "m"


def test_a_folder_hash_follows_contents_and_names_but_not_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "inference.pdiparams").write_bytes(b"weights")
    before = folder_sha256(tmp_path)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "lock").write_bytes(b"anything")
    assert folder_sha256(tmp_path) == before
    (tmp_path / "inference.pdiparams").write_bytes(b"other weights")
    assert folder_sha256(tmp_path) != before


def test_a_missing_package_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise paddle.PackageNotFoundError(name)

    monkeypatch.setattr(paddle, "version", missing)
    assert installed("paddlepaddle", "paddlepaddle-gpu") == "paddlepaddle not installed"
