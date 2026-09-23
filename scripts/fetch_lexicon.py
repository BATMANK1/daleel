#!/usr/bin/env python3
"""Download and verify the CAMeL MSA frequency list used as the gate's lexicon.

    python3 scripts/fetch_lexicon.py

Downloads the zip (69 MB) into data/external/, verifies its checksum against
the pinned version, and checks once that the whole list is sorted by
descending frequency, which the loader relies on to stop reading early. Then
reports how many words survive a range of cutoffs.

The list is by the CAMeL Lab at NYU Abu Dhabi and is licensed CC BY-SA 4.0,
so it is downloaded rather than committed. Cite it as:
Khalifa, Inoue, Alhafni, Baimukan, Bouamor and Habash (2021),
Camel Arabic Frequency Lists,
https://github.com/CAMeL-Lab/Camel_Arabic_Frequency_Lists
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile

from daleel.ingest.lexicon import LEXICON_MEMBER, LEXICON_ZIP, read_frequency_list

URL = (
    "https://github.com/CAMeL-Lab/Camel_Arabic_Frequency_Lists"
    "/releases/download/v1.0/MSA_freq_lists.tsv.zip"
)
SHA256 = "99a6792501af8f1a8ed5209bfada961ae4fc076823e62b16883ef2e4c39b39a4"
CUTOFFS = (100_000, 10_000, 1_000, 100, 10)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    LEXICON_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if LEXICON_ZIP.exists():
        print(f"{LEXICON_ZIP} already present")
    else:
        partial = LEXICON_ZIP.with_suffix(".part")
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(LEXICON_ZIP)

    actual = sha256_of(LEXICON_ZIP)
    if actual != SHA256:
        print(f"error: checksum {actual} does not match the pinned {SHA256}", file=sys.stderr)
        print("The upstream file may have changed. Delete it and investigate.", file=sys.stderr)
        return 1
    print("checksum matches the pinned version")

    counts = dict.fromkeys(CUTOFFS, 0)
    total = 0
    with zipfile.ZipFile(LEXICON_ZIP) as archive, archive.open(LEXICON_MEMBER) as raw:
        lines = io.TextIOWrapper(raw, encoding="utf-8")
        # min_frequency=0 reads every line, so the sort check covers the whole file.
        for _, frequency in read_frequency_list(lines, min_frequency=0):
            total += 1
            for cutoff in CUTOFFS:
                if frequency >= cutoff:
                    counts[cutoff] += 1

    print(f"whole list sorted by descending frequency: {total:,} words")
    for cutoff in CUTOFFS:
        print(f"  frequency >= {cutoff:>7,}: {counts[cutoff]:>10,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
