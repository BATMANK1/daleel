"""Keep what an OCR engine reads, so that no page is rendered or read twice.

A page read with dots.mocr on this project's GPU takes two to four minutes, and
the covers of four documents, drawn with hundreds of shadings, take minutes
just to render at 200 DPI. So a second run of the pipeline must neither render
nor read a page it has read before.

Each reading is kept under a key made of the engine's tag and everything that
went into the image the engine read: the PDF's SHA-256, the page, the
resolution, pypdfium2's version and daleel.ocr.render's RENDERING. The tag must
name every setting that changes what the engine reads, and dots.mocr's does.
The PDF is identified by its content, so a copy under another name is the same
document, and a document changed in any way is a new one.

Each reading is a JSON file under data/interim/ocr_cache/, written whole or not
at all, so an interrupted run leaves nothing half-written behind. An answer cut
off at a length limit is not kept, since a larger limit could finish it. Like
the rest of data/interim/, the cache stays out of git.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from daleel.ocr.engine import Block, Engine, Result
from daleel.ocr.render import RENDERING, render_page

CACHE = Path("data/interim/ocr_cache")


@dataclass(frozen=True)
class Reading:
    """One page as an engine read it, the page's size in points, and whether it was kept."""

    result: Result
    page_size: tuple[float, float]
    cached: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_from_json(data: dict[str, Any]) -> Result:
    blocks = tuple(
        Block(
            category=block["category"],
            text=block["text"],
            box=None if block["box"] is None else tuple(block["box"]),
            html=block["html"],
        )
        for block in data["blocks"]
    )
    return Result(
        text=data["text"],
        seconds=data["seconds"],
        warnings=data["warnings"],
        response=data["response"],
        blocks=blocks,
        truncated=data["truncated"],
    )


class CachedReader:
    """Renders and reads pages with an engine, keeping every reading to read back later."""

    def __init__(self, engine: Engine, folder: Path = CACHE, *, dpi: int = 200) -> None:
        if dpi < 1:
            raise ValueError(f"a resolution must be positive, not {dpi}")
        self.engine = engine
        self.folder = folder
        self.dpi = dpi
        self.renderer = f"pypdfium2 {version('pypdfium2')}, rendering {RENDERING}"
        self._tag = engine.tag()
        self._documents: dict[Path, str] = {}
        # Pages read back from the cache, and pages the engine read.
        self.hits = 0
        self.misses = 0

    def tag(self) -> str:
        return self._tag

    def describe(self) -> str:
        return self.engine.describe()

    def key(self, path: Path, page: int) -> tuple[str, dict[str, Any]]:
        """The key of one page's reading, and what it is made of."""
        resolved = path.resolve()
        if resolved not in self._documents:
            self._documents[resolved] = file_sha256(resolved)
        material = {
            "engine": self._tag,
            "pdf": self._documents[resolved],
            "page": page,
            "dpi": self.dpi,
            "renderer": self.renderer,
        }
        key = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        return key, material

    def entry(self, key: str) -> Path:
        # A level of folders by the key's first two characters keeps any one
        # folder small.
        return self.folder / key[:2] / f"{key}.json"

    def read(self, path: Path, page: int) -> Reading:
        """One page, numbered from 1, from the cache or else rendered and read."""
        key, material = self.key(path, page)
        entry = self.entry(key)
        kept = self._load(entry, material)
        if kept is not None:
            self.hits += 1
            return kept
        rendered = render_page(path, page, dpi=self.dpi)
        result = self.engine.recognize(rendered.png())
        self.misses += 1
        if not result.truncated:
            self._store(
                entry, {**material, "page_size": list(rendered.page_size), **asdict(result)}
            )
        return Reading(result=result, page_size=rendered.page_size, cached=False)

    @staticmethod
    def _load(entry: Path, material: dict[str, Any]) -> Reading | None:
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            if any(data[name] != value for name, value in material.items()):
                return None
            width, height = data["page_size"]
            return Reading(result=result_from_json(data), page_size=(width, height), cached=True)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            # A damaged entry is read again, and replaced.
            return None

    @staticmethod
    def _store(entry: Path, data: dict[str, Any]) -> None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=entry.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False)
            # A rename replaces the entry in one step, so no reader sees half of one.
            Path(name).replace(entry)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
