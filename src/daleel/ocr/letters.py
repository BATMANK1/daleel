"""Repair the letters of other scripts a model writes inside Arabic words.

dots.mocr, as served here, writes two Arabic spellings with letters of other
scripts, the same way each time. The word for a week comes back with the Thai
letter SO SUA standing for its sin and ba: on the library deck's page 8, and
three times on the organizational regulations' pages 21 and 33 read for T2. Ta
came back once as the Hebrew letter tav, on the library deck's page 4: the same
character with its first UTF-8 byte one lower. Reading those blocks again, on
their own or with the whole page, does not mend them (eval/results/layout.md),
so they are repaired here. No page in this corpus prints Thai or Hebrew, and a
repair is made only inside an Arabic word: next to an Arabic letter.

Any other letter of another script inside an Arabic word, or a character the
model could not write at all, is left as read and reported.
"""

from __future__ import annotations

import unicodedata

# Each letter seen in place of Arabic, and the Arabic it stands for.
REPAIRS = {
    "\u0e2a": "\u0633\u0628",  # THAI CHARACTER SO SUA, for sin and ba
    "\u05ea": "\u062a",  # HEBREW LETTER TAV, for ta
}

# What a decoder writes for bytes that are not UTF-8.
REPLACEMENT = "\ufffd"


def _arabic_letter(char: str) -> bool:
    return "\u0621" <= char <= "\u063a" or "\u0641" <= char <= "\u064a"


def _foreign(char: str) -> bool:
    """A letter of a script other than Arabic or Latin, or a character that could not be written."""
    if char == REPLACEMENT:
        return True
    if not char.isalpha():
        return False
    return unicodedata.name(char, "").split(" ")[0] not in {"ARABIC", "LATIN"}


def _word(text: str, index: int) -> str:
    """The run of letters around text[index], as read."""
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == REPLACEMENT):
        start -= 1
    end = index + 1
    while end < len(text) and (text[end].isalpha() or text[end] == REPLACEMENT):
        end += 1
    return text[start:end]


def repair_letters(text: str) -> tuple[str, list[str]]:
    """The text with each known letter repaired inside Arabic words, and a note on each found."""
    notes: list[str] = []
    repaired: list[str] = []
    for index, char in enumerate(text):
        beside = (index > 0 and _arabic_letter(text[index - 1])) or (
            index + 1 < len(text) and _arabic_letter(text[index + 1])
        )
        if beside and _foreign(char):
            name = unicodedata.name(char, f"U+{ord(char):04X}")
            word = _word(text, index)
            if char in REPAIRS:
                notes.append(f"{name} in {word} read as {REPAIRS[char]}")
                repaired.append(REPAIRS[char])
                continue
            notes.append(f"{name} in {word} left as read")
        repaired.append(char)
    return "".join(repaired), notes
