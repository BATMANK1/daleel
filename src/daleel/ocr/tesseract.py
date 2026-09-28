"""Run Tesseract on rendered pages, recording what produced each result.

Tesseract runs as its command-line program, reading a PNG on stdin and writing
text to stdout, so no Python binding sits between the engine and its output.

The same engine version gives different text with different model files: the
standard, fast and best Arabic models differ, and every distribution packages
its own. A result is only reproducible with the model recorded, so each run
records the version and the SHA-256 of the traineddata file it used.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

BINARY = "tesseract"

# First line of `tesseract --list-langs`: List of available languages in "/path/" (3):
_TESSDATA = re.compile(r'"(?P<path>[^"]+)"')


class TesseractError(RuntimeError):
    """Tesseract exited with an error."""


@dataclass(frozen=True)
class Settings:
    """Everything passed to Tesseract that can change its output."""

    lang: str = "ara"
    # Page segmentation mode 3 is Tesseract's default: fully automatic layout.
    psm: int = 3
    # Stated outright, since Tesseract otherwise trusts the image's metadata or
    # guesses, and a wrong guess changes how it scales the text.
    dpi: int = 300

    def command(self, binary: str = BINARY) -> list[str]:
        """The command line that reads a PNG on stdin and writes text to stdout."""
        return [
            binary,
            "stdin",
            "stdout",
            "-l",
            self.lang,
            "--psm",
            str(self.psm),
            "--dpi",
            str(self.dpi),
        ]


DEFAULT = Settings()


@dataclass(frozen=True)
class Result:
    """What Tesseract read from one image, how long it took, and what it warned about."""

    text: str
    seconds: float
    warnings: str


def _run(command: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, input=stdin, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise TesseractError(f"{command[0]} is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise TesseractError(f"{command[0]} failed: {message}") from exc


def recognize(png: bytes, settings: Settings = DEFAULT, binary: str = BINARY) -> Result:
    """Read the text in one PNG image."""
    start = time.perf_counter()
    completed = _run(settings.command(binary), stdin=png)
    seconds = time.perf_counter() - start
    return Result(
        text=completed.stdout.decode("utf-8"),
        seconds=seconds,
        warnings=completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def parse_version(output: str) -> str:
    """The version number from `tesseract --version`, whose first line is "tesseract 5.3.4"."""
    first = output.strip().splitlines()[0] if output.strip() else ""
    name, _, number = first.partition(" ")
    if name != "tesseract" or not number:
        raise TesseractError(f"unexpected version output: {first!r}")
    return number.strip()


def parse_tessdata(output: str) -> Path:
    """The model directory named on the first line of `tesseract --list-langs`."""
    match = _TESSDATA.search(output)
    if match is None:
        raise TesseractError(f"no model directory in: {output.strip()[:200]!r}")
    return Path(match["path"])


def version(binary: str = BINARY) -> str:
    """The installed Tesseract's version number."""
    completed = _run([binary, "--version"])
    # Older releases print the version on stderr.
    return parse_version(completed.stdout.decode() or completed.stderr.decode())


def model_hashes(lang: str, binary: str = BINARY) -> dict[str, str]:
    """The SHA-256 of each traineddata file Tesseract loads for a language setting.

    A setting such as ara+eng loads one model per language, so each is hashed.
    """
    completed = _run([binary, "--list-langs"])
    folder = parse_tessdata(completed.stdout.decode() or completed.stderr.decode())
    return {
        name: hashlib.sha256((folder / f"{name}.traineddata").read_bytes()).hexdigest()
        for name in lang.split("+")
    }
