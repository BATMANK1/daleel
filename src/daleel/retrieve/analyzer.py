"""Arabic text as the terms a lexical index matches on.

A question and a page rarely spell a word alike. The page writes الطالبُ with
its vowel, the question الطالب; the page says للطلاب where the question says
الطلاب. The analyzer makes such pairs one term in three steps, each of which
can be turned off, so what each buys can be measured:

- normalization for comparison (daleel.normalize.arabic), which folds the
  spellings of one word together: alef forms, taa marbuta, diacritics,
  tatweel, digits in either script;
- stopwords: function words, the question words a student opens with, and
  the colloquial ones of Saudi questions (وش, ايش, ابغى), which say nothing
  about what a page holds;
- light stemming, as Lucene's Arabic stemmer does it, after the light
  stemmer of Larkey, Ballesteros and Connell: one prefix, the article or a
  conjunction with it (PREFIXES), then common suffixes (SUFFIXES), each only
  where enough of the word is left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from daleel.normalize.arabic import for_comparison

_WORD = re.compile(r"\w+")


def _normalized(text: str) -> frozenset[str]:
    return frozenset(for_comparison(word) for word in text.split())


STOPWORDS = _normalized(
    """
    من إلى الى عن على في مع و أو او ثم بل لكن أن إن ان كان كانت يكون تكون هذا هذه
    ذلك تلك هؤلاء الذي التي الذين ما ماذا لا لم لن ليس قد كل بعض أي اي غير هو هي هم
    أنا انا نحن أنت انت كما حيث إذا اذا عند عندما حتى بين لدى له لها لهم به بها فيه
    فيها منه منها عليه عليها إليه اليه هل كيف متى أين اين لماذا كم لو لي
    وش ايش شو ليش وين ابغى ابي ابغي أبغى أبي طيب يعني احنا اقدر أقدر عشان
    """
)

# Lucene's ArabicStemmer, on text already normalized, so taa marbuta is haa:
# the article and its conjunctions, and then the endings of possession,
# number and relation.
ALEF, HEH = chr(0x0627), chr(0x0647)
PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل", "و")
SUFFIXES = (HEH + ALEF, "ان", "ات", "ون", "ين", "يه", HEH, "ي")


def light_stem(word: str) -> str:
    """A word with one prefix and its suffixes taken off, where enough is left.

    A prefix comes off only where two letters remain after it, and the single
    letter و only from a word of four letters or more. Each suffix in turn
    comes off where two letters remain.
    """
    for prefix in PREFIXES:
        if len(prefix) == 1 and len(word) < 4:
            continue
        if word.startswith(prefix) and len(word) >= len(prefix) + 2:
            word = word[len(prefix) :]
            break
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) >= len(suffix) + 2:
            word = word[: -len(suffix)]
    return word


@dataclass(frozen=True)
class Analyzer:
    """Turns text into terms: words, normalized, without stopwords, stemmed."""

    normalize: bool = True
    stopwords: bool = True
    stem: bool = True

    def terms(self, text: str) -> list[str]:
        if self.normalize:
            text = for_comparison(text)
        found = _WORD.findall(text.lower())
        if self.stopwords:
            found = [word for word in found if word not in STOPWORDS]
        if self.stem:
            found = [light_stem(word) for word in found]
        return found
