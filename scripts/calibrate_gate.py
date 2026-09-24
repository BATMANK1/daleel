#!/usr/bin/env python3
"""Reproduce every number in eval/results/gate_calibration.md. No thresholds.

    python3 scripts/calibrate_gate.py

Four sections, each computed from the PDFs, the ground truth and the lexicon:

1. Backends: how well each extractor's text matches the ground truth, both the
   Arabic words and every run of digits or Latin letters the page prints.
2. Proxies and truth: everything daleel.ingest.quality measures, beside
   precision (the share of the layer's words that really appear on the page)
   and recall (the share of the page's words the layer captured). Both compare
   bags of normalized Arabic words, so reading order does not matter.
3. Token validity by lexicon cutoff, for the layer and for the ground truth
   (the ceiling), and how well each cutoff separates sound pages from broken.
4. Lexicon noise: how often the extraction errors found during transcription
   occur in the raw frequency list.

Ground truth never leaves this machine, so this runs locally and only its
numbers are published.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pdfplumber

from daleel.ingest.extract import page_text
from daleel.ingest.lexicon import LEXICON_MEMBER, LEXICON_ZIP, load_lexicon, read_frequency_list
from daleel.ingest.metadata import read_metadata
from daleel.ingest.quality import arabic_tokens, measure, single_letter_share, token_validity
from daleel.ingest.router import route
from daleel.normalize.arabic import for_comparison

RAW = Path("data/raw")
GROUND_TRUTH = Path("data/interim/ground_truth")
PAGES = (
    ("guidance_manual", 2),
    ("student_guide_2025", 30),
    ("library_services_2024_2025", 10),
    ("orientation_1446", 3),
    ("academic_weeks_1448", 1),
)
CUTOFFS = (10_000, 1_000, 100)
# For calibration only: a page whose layer matches its ground truth this well is
# "sound". Measured pages sit far from it on both sides (99%+ against 50% or less).
SOUND_PRECISION = 0.9
# Extraction errors found while transcribing, and the words they should have been.
KNOWN_ERRORS = {
    "\u0648\u0624\u064a\u062a\u0646\u0627": "guidance manual layer, for the word below",
    "\u0631\u0624\u064a\u062a\u0646\u0627": "the correct word",
    "\u0645\u0635\u0627\u062f\u0631\u0627\u0644\u0645\u0639\u0644\u0648\u0645\u0627\u062a": (
        "printed run together on the library slide"
    ),
}
# A run starts and ends with a letter, digit or %, so punctuation that bidi display
# moves around a list number ("1." extracted as ".1") is not mistaken for reversal.
LATIN_RUN = re.compile(r"[0-9A-Za-z](?:[0-9A-Za-z%./:\-]*[0-9A-Za-z%])?")
BRACKETED = re.compile(r"\([^()\s]+\)")


def ground_truth_text(doc: str, page: int) -> str:
    """The page's transcription: its text file plus, for tables, every CSV cell."""
    base = GROUND_TRUTH / f"{doc}_p{page:02d}"
    parts = []
    if base.with_suffix(".txt").exists():
        parts.append(base.with_suffix(".txt").read_text(encoding="utf-8"))
    if base.with_suffix(".csv").exists():
        with base.with_suffix(".csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                parts.extend(row.values())
    return "\n".join(parts)


def _plumber(doc: str, page: int, **options: str) -> str:
    with pdfplumber.open(RAW / f"{doc}.pdf") as pdf:
        return pdf.pages[page - 1].extract_text(**options) or ""


def _pdftotext(doc: str, page: int) -> str:
    command = ["pdftotext", "-f", str(page), "-l", str(page), str(RAW / f"{doc}.pdf"), "-"]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


BACKENDS: dict[str, Callable[[str, int], str]] = {
    "pdfplumber": _plumber,
    "pdfplumber-rtl": lambda doc, page: _plumber(doc, page, char_dir_render="rtl"),
    "pypdfium2": lambda doc, page: page_text(RAW / f"{doc}.pdf", page),
}
if shutil.which("pdftotext"):
    BACKENDS["pdftotext"] = _pdftotext


def overlap(layer: list[str], truth: list[str]) -> tuple[float | None, float | None]:
    """Precision and recall of the layer's words against the page's, as bags."""
    common = sum((Counter(layer) & Counter(truth)).values())
    precision = common / len(layer) if layer else None
    recall = common / len(truth) if truth else None
    return precision, recall


def runs_survive(truth: str, text: str) -> tuple[int, int, int, int]:
    """Of the page's distinct digit and Latin runs: total, intact, reversed, missing.

    Single characters and palindromes are skipped: reversing them changes nothing.
    """
    runs = {r for r in LATIN_RUN.findall(truth) if len(r) > 1 and r != r[::-1]}
    intact = sum(r in text for r in runs)
    backwards = sum(r not in text and r[::-1] in text for r in runs)
    return len(runs), intact, backwards, len(runs) - intact - backwards


def pct(value: float | None) -> str:
    return "   -" if value is None else f"{value:4.0%}"


def versions() -> str:
    found = [f"{p} {importlib.metadata.version(p)}" for p in ("pypdfium2", "pdfplumber")]
    if shutil.which("pdftotext"):
        banner = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr
        found.append(banner.splitlines()[0] if banner else "pdftotext")
    return ", ".join(found)


def section_backends() -> None:
    print("1. BACKENDS: Arabic words against ground truth; digit and Latin runs as printed")
    print(
        f"{'page':32} {'backend':15} {'prec':>4} {'rec':>4} | "
        f"{'runs':>4} {'intact':>6} {'reversed':>8} {'missing':>7} | {'(bracketed) exact':>17}"
    )
    for doc, page in PAGES:
        truth_text = ground_truth_text(doc, page)
        truth = arabic_tokens(for_comparison(truth_text))
        bracketed = set(BRACKETED.findall(truth_text))
        for name, extract in BACKENDS.items():
            text = for_comparison(extract(doc, page))
            precision, recall = overlap(arabic_tokens(text), truth)
            total, intact, backwards, missing = runs_survive(truth_text, text)
            exact = sum(b in text for b in bracketed)
            print(
                f"{doc + f'_p{page:02d}':32} {name:15} {pct(precision)} {pct(recall)} | "
                f"{total:>4} {intact:>6} {backwards:>8} {missing:>7} | "
                f"{f'{exact}/{len(bracketed)}':>17}"
            )
        print()


def section_proxies(lexicons: dict[int, frozenset[str]]) -> list[tuple]:
    print("2. PROXIES AND TRUTH, pypdfium2")
    print(
        f"{'page':32} {'route':10} {'content':>7} {'presf':>5} {'cid':>4} {'fffd':>4} "
        f"{'tok':>4} {'gt':>4} {'1-let':>5} {'gt1-l':>5} | {'prec':>4} {'rec':>4}"
    )
    rows = []
    for doc, page in PAGES:
        meta = read_metadata(RAW / f"{doc}.pdf")
        predicted = route(producer=meta.producer, creator=meta.creator).path.value
        raw = page_text(RAW / f"{doc}.pdf", page)
        quality = measure(raw, lexicons[CUTOFFS[-1]])
        layer = arabic_tokens(for_comparison(raw))
        truth = arabic_tokens(for_comparison(ground_truth_text(doc, page)))
        precision, recall = overlap(layer, truth)
        name = f"{doc}_p{page:02d}"
        print(
            f"{name:32} {predicted:10} {quality.content_chars:>7} "
            f"{pct(quality.presform_ratio):>5} {quality.cid_placeholders:>4} "
            f"{quality.replacement_chars:>4} {len(layer):>4} {len(truth):>4} "
            f"{pct(single_letter_share(layer)):>5} {pct(single_letter_share(truth)):>5} | "
            f"{pct(precision):>4} {pct(recall):>4}"
        )
        rows.append(
            (
                name,
                precision is not None and precision >= SOUND_PRECISION,
                [token_validity(layer, lexicons[c]) for c in CUTOFFS],
                [token_validity(truth, lexicons[c]) for c in CUTOFFS],
            )
        )
    print()
    return rows


def section_cutoffs(rows: list[tuple]) -> None:
    print("3. TOKEN VALIDITY by lexicon cutoff: text layer, then ground truth (the ceiling)")
    header = "  ".join(f"{c:>6,}" for c in CUTOFFS)
    print(f"{'page':32} layer: {header}   truth: {header}")
    for name, _, layer_scores, truth_scores in rows:
        layer_cells = "  ".join(f"{pct(s):>6}" for s in layer_scores)
        truth_cells = "  ".join(f"{pct(s):>6}" for s in truth_scores)
        print(f"{name:32}        {layer_cells}          {truth_cells}")
    print()
    print("   separation: worst sound page minus best broken page (higher is better)")
    for i, cutoff in enumerate(CUTOFFS):
        sound = [row[2][i] for row in rows if row[1] and row[2][i] is not None]
        broken = [row[2][i] for row in rows if not row[1] and row[2][i] is not None]
        if sound and broken:
            gap = min(sound) - max(broken)
            print(
                f"   cutoff {cutoff:>6,}: sound >= {min(sound):.0%}, "
                f"broken <= {max(broken):.0%}, gap {gap:+.0%}"
            )
    print()


def section_lexicon_noise() -> None:
    print("4. LEXICON NOISE: known extraction errors in the raw frequency list")
    found = dict.fromkeys(KNOWN_ERRORS, 0)
    with zipfile.ZipFile(LEXICON_ZIP) as archive, archive.open(LEXICON_MEMBER) as raw:
        lines = io.TextIOWrapper(raw, encoding="utf-8")
        for word, frequency in read_frequency_list(lines, min_frequency=0):
            if word in found:
                found[word] = frequency
    for word, note in KNOWN_ERRORS.items():
        escaped = word.encode("unicode_escape").decode()
        print(f"   {found[word]:>7,}  {escaped}  ({note})")


def main() -> int:
    print(f"versions: {versions()}\n")
    section_backends()
    lexicons = {cutoff: load_lexicon(min_frequency=cutoff) for cutoff in CUTOFFS}
    rows = section_proxies(lexicons)
    section_cutoffs(rows)
    section_lexicon_noise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
