"""Arabic normalization for comparing text, not for displaying it.

After normalization, two strings a reader would call the same word compare
equal, however the PDF happened to encode them. The core is the light
normalization long used in Arabic information retrieval, the same set Lucene's
Arabic analyzer applies: alefs carrying hamza become a bare alef, alef maqsura
becomes yaa, taa marbuta becomes haa, and tatweel and diacritics are removed.
Two steps are added for this corpus: folding presentation forms back to base
letters, and removing invisible characters.

Several steps merge letters a careful reader keeps apart, such as taa marbuta
and haa. That is acceptable for matching words and wrong for display or for
ground truth, which records the page exactly. Each step is its own function so
a caller can choose a different combination.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

# The contextual glyph variants a renderer picks per letter position. A text
# layer that stores these has stored display forms rather than characters.
PRESENTATION_FORM_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Within those, the presentation forms of the harakat themselves. NFKC turns an
# isolated form into a space plus the mark, and the space must not survive.
MARK_FORM_RANGE = (0xFE70, 0xFE7F)

# Explicit bidirectional formatting characters: meaningful to a renderer, noise
# to a comparison. The inventory counts them; normalization removes them.
BIDI_CONTROLS = frozenset(
    {
        0x061C,  # Arabic letter mark
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x202A,  # left-to-right embedding
        0x202B,  # right-to-left embedding
        0x202C,  # pop directional formatting
        0x202D,  # left-to-right override
        0x202E,  # right-to-left override
        0x2066,  # left-to-right isolate
        0x2067,  # right-to-left isolate
        0x2068,  # first strong isolate
        0x2069,  # pop directional isolate
    }
)
# Everything a comparison should not see: the bidi controls, the zero-width
# characters, and the byte order mark.
INVISIBLES = BIDI_CONTROLS | frozenset(
    {
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0xFEFF,  # byte order mark
    }
)
LAYOUT_WHITESPACE = frozenset("\t\n\r")

TATWEEL = "\u0640"
DIACRITIC_RANGES = (
    (0x064B, 0x065F),  # harakat, tanween, shadda, sukun and related marks
    (0x0670, 0x0670),  # superscript alef
    (0x06D6, 0x06ED),  # Quranic annotation marks
)
BARE_ALEF = "\u0627"
ALEF_VARIANTS = "\u0622\u0623\u0625\u0671"  # alef with madda, hamza above, hamza below, wasla
ALEF_MAQSURA, YAA = "\u0649", "\u064a"
TAA_MARBUTA, HAA = "\u0629", "\u0647"

_ALEF_TABLE = str.maketrans(ALEF_VARIANTS, BARE_ALEF * len(ALEF_VARIANTS))


def in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Whether a codepoint falls inside any of the inclusive ranges."""
    return any(low <= codepoint <= high for low, high in ranges)


def strip_invisible(text: str) -> str:
    """Remove bidi controls, zero-width characters and stray control codes.

    Tabs and line breaks are layout, not noise, so they stay.
    """
    return "".join(
        ch
        for ch in text
        if ord(ch) not in INVISIBLES
        and not (unicodedata.category(ch) == "Cc" and ch not in LAYOUT_WHITESPACE)
    )


def fold_presentation_forms(text: str) -> str:
    """Replace each presentation-form glyph with the base letters it shows."""
    folded = []
    for ch in text:
        codepoint = ord(ch)
        if not in_ranges(codepoint, PRESENTATION_FORM_RANGES):
            folded.append(ch)
            continue
        base = unicodedata.normalize("NFKC", ch)
        if MARK_FORM_RANGE[0] <= codepoint <= MARK_FORM_RANGE[1]:
            base = base.replace(" ", "")
        folded.append(base)
    return "".join(folded)


def remove_tatweel(text: str) -> str:
    """Remove the stroke that stretches letters for justification."""
    return text.replace(TATWEEL, "")


def strip_diacritics(text: str) -> str:
    """Remove harakat, tanween, shadda, sukun, superscript alef and Quranic marks."""
    return "".join(ch for ch in text if not in_ranges(ord(ch), DIACRITIC_RANGES))


def unify_alef(text: str) -> str:
    """Turn alef with madda, with hamza above or below, and wasla into a bare alef."""
    return text.translate(_ALEF_TABLE)


def unify_alef_maqsura(text: str) -> str:
    """Turn alef maqsura into yaa."""
    return text.replace(ALEF_MAQSURA, YAA)


def unify_taa_marbuta(text: str) -> str:
    """Turn taa marbuta into haa."""
    return text.replace(TAA_MARBUTA, HAA)


def collapse_whitespace(text: str) -> str:
    """Join all tokens with single spaces, removing line breaks too."""
    return " ".join(text.split())


# Order matters: invisibles go before folding so they cannot sit inside a
# ligature, and folding runs before the tatweel and diacritic steps because
# some presentation forms expand into exactly those characters.
COMPARISON_STEPS: tuple[Callable[[str], str], ...] = (
    strip_invisible,
    fold_presentation_forms,
    remove_tatweel,
    strip_diacritics,
    unify_alef,
    unify_alef_maqsura,
    unify_taa_marbuta,
    collapse_whitespace,
)


def for_comparison(text: str) -> str:
    """Normalize text so that the same words compare equal however encoded."""
    for step in COMPARISON_STEPS:
        text = step(text)
    return text
