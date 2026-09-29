"""Run PaddleOCR on rendered pages, and return its text in Arabic reading order.

PaddleOCR is an optional dependency, installed with the paddle extra
(pip install -e ".[paddle]"). It is imported only when an engine is built, so
the rest of the package, and its tests, run without it.

The models are named outright rather than chosen from a language code, since
PaddleOCR's mapping from languages to models changes between releases: here
PP-OCRv5's server detector and its Arabic-script recognizer. The page
preprocessing PaddleOCR can apply (orientation classification, unwarping and
text-line orientation) is switched off: the pages are digital renders, upright
and flat, so those models could only add errors.

The recognizer reads a line image left to right, in visual order, and PaddleX
converts each line to reading order with python-bidi before returning it. Its
Hugging Face backend skips that step, so the backend is fixed as well, and
python-bidi's version is recorded beside PaddleOCR's. Joining the lines into a
page is left to daleel.ocr.reading_order, since PaddleOCR sorts a row's boxes
left to right.

PaddlePaddle is pinned at 3.2.2. Versions 3.3.0 and 3.3.1 fail on MKL-DNN,
the default CPU path, and without MKL-DNN the server detector allocates about
5 GB per megapixel of page: 58 GB for the academic calendar at 300 DPI.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image

from daleel.ocr.engine import Result
from daleel.ocr.reading_order import Box, arabic_reading_order


@dataclass(frozen=True)
class Settings:
    """Everything passed to PaddleOCR that can change its output or its speed."""

    detection_model: str = "PP-OCRv5_server_det"
    recognition_model: str = "arabic_PP-OCRv5_mobile_rec"
    device: str = "cpu"
    # PaddleOCR's own defaults, stated so that a change to them shows.
    cpu_threads: int = 10
    enable_mkldnn: bool = True
    # Lines the recognizer scores below this are dropped. PaddleOCR keeps
    # everything by default, including what it reads in background graphics.
    min_score: float = 0.0
    # The page is widened by this factor before OCR, so every line image reaches
    # the recognizer wider and it has more steps per letter (PaddleOCR issue
    # 18349). 1 leaves the page as rendered.
    stretch: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_score <= 1:
            raise ValueError(f"min_score is a recognizer score from 0 to 1, not {self.min_score}")
        if self.stretch <= 0:
            raise ValueError(f"stretch must be positive, not {self.stretch}")

    def options(self) -> dict[str, object]:
        """The keyword arguments for paddleocr.PaddleOCR."""
        return {
            "text_detection_model_name": self.detection_model,
            "text_recognition_model_name": self.recognition_model,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_rec_score_thresh": self.min_score,
            "engine": "paddle_static",
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "enable_mkldnn": self.enable_mkldnn,
        }


DEFAULT = Settings()


def installed(*names: str) -> str:
    """The first of these packages that is installed, with its version.

    PaddlePaddle is published as paddlepaddle for CPUs and as paddlepaddle-gpu
    for CUDA, so either name may hold it.
    """
    for name in names:
        try:
            return f"{name} {version(name)}"
        except PackageNotFoundError:
            continue
    return f"{names[0]} not installed"


def model_folder(name: str) -> Path:
    """Where PaddleX keeps an official model once it has downloaded it."""
    cache = os.environ.get("PADDLE_PDX_CACHE_HOME") or Path.home() / ".paddlex"
    return Path(cache) / "official_models" / name


def folder_sha256(folder: Path) -> str:
    """One SHA-256 over every file in a folder, each with its path, hidden files aside."""
    if not folder.is_dir():
        raise FileNotFoundError(f"no model in {folder}")
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            digest.update(relative.as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def stretched(png: bytes, factor: float) -> bytes:
    """The page widened by a factor, its height unchanged. A factor of 1 returns it as it is."""
    if factor == 1:
        return png
    with Image.open(io.BytesIO(png)) as image:
        wider = image.resize((round(image.width * factor), image.height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    wider.save(buffer, format="PNG")
    return buffer.getvalue()


class PaddleEngine:
    """PaddleOCR with fixed models and settings.

    `ocr` stands in for paddleocr.PaddleOCR in tests; anything with PaddleOCR's
    predict method will do.
    """

    def __init__(self, settings: Settings = DEFAULT, ocr: Any = None) -> None:
        self.settings = settings
        if ocr is None:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(**settings.options())
        self._ocr = ocr
        self.packages = [
            installed("paddleocr"),
            installed("paddlex"),
            installed("paddlepaddle", "paddlepaddle-gpu"),
            installed("python-bidi"),
        ]
        # PaddleOCR downloads its models when it is built, so they exist by now.
        self.models = {
            name: folder_sha256(model_folder(name))
            for name in (settings.detection_model, settings.recognition_model)
        }

    def tag(self) -> str:
        settings = self.settings
        paddleocr = self.packages[0].split()[-1]
        models = "+".join(sha[:8] for sha in self.models.values())
        tag = f"paddleocr-{paddleocr}-{models}-{settings.device.replace(':', '')}"
        if settings.min_score:
            tag += f"-min{settings.min_score:g}"
        if settings.stretch != 1:
            tag += f"-stretch{settings.stretch:g}"
        return tag

    def describe(self) -> str:
        models = ", ".join(f"{name} sha256 {sha[:16]}" for name, sha in self.models.items())
        settings = self.settings
        mkldnn = "on" if settings.enable_mkldnn else "off"
        return (
            f"{', '.join(self.packages)}; {models}; device {settings.device}, "
            f"{settings.cpu_threads} CPU threads, MKL-DNN {mkldnn}; "
            f"min score {settings.min_score:g}, stretch {settings.stretch:g}"
        )

    def recognize(self, png: bytes) -> Result:
        # PaddleX reads a path with OpenCV, which gives the BGR pixels its models
        # expect; an array from Pillow would reach them as RGB, colours swapped.
        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "page.png"
            start = time.perf_counter()
            page.write_bytes(stretched(png, self.settings.stretch))
            results = list(self._ocr.predict(str(page)))
            seconds = time.perf_counter() - start
        if len(results) != 1:
            raise RuntimeError(f"PaddleOCR returned {len(results)} results for one page")
        (result,) = results
        boxes = [
            Box.around(text, polygon)
            for text, polygon in zip(result["rec_texts"], result["rec_polys"], strict=True)
        ]
        return Result(text=arabic_reading_order(boxes), seconds=seconds)
