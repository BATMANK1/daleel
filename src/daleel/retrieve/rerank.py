"""Reranking: the first chunks of a ranking reordered by a cross-encoder.

A cross-encoder reads a question and a chunk together and scores how well the
chunk answers it, which is slower than comparing vectors and usually sharper.
So it reorders only the first RERANK_DEPTH chunks of a fused ranking. The
model is BAAI's bge-reranker-v2-m3, multilingual like the encoders.

As with the encoders' vectors, its scores are computed once on a machine that
has the model, by scripts/score_rerank_pool.py, and kept in
data/interim/rerank/<name>.json under the hash of the question and the chunk's
text, so that an evaluation can try any depth and any fusion without the model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RERANK = Path("data/interim/rerank")
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANKER_NAME = "bge-reranker-v2-m3"
RERANK_DEPTH = 20
MAX_TOKENS = 512


def pair_key(question: str, text: str) -> str:
    """The key a question and a chunk's text are scored under."""
    return hashlib.sha256(f"{question}\0{text}".encode()).hexdigest()[:24]


@dataclass
class ScoreStore:
    """Cross-encoder scores of questions and chunks, and a note of what made them."""

    scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def missing(self, question: str, texts: Sequence[str]) -> list[str]:
        """The texts, each once, with no score for the question yet."""
        absent = (text for text in texts if pair_key(question, text) not in self.scores)
        return list(dict.fromkeys(absent))

    def add(self, question: str, texts: Sequence[str], scores: Sequence[float]) -> None:
        if len(texts) != len(scores):
            raise ValueError(f"{len(texts)} texts but {len(scores)} scores")
        for text, score in zip(texts, scores, strict=True):
            self.scores[pair_key(question, text)] = float(score)

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        absent = self.missing(question, texts)
        if absent:
            raise KeyError(
                f"{len(absent)} chunks have no score for this question; score a deeper "
                "pool with scripts/score_rerank_pool.py"
            )
        return [self.scores[pair_key(question, text)] for text in texts]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"meta": self.meta, "scores": dict(sorted(self.scores.items()))}
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ScoreStore:
        body = json.loads(path.read_text(encoding="utf-8"))
        return cls({key: float(value) for key, value in body["scores"].items()}, body["meta"])


Scorer = Callable[[str, Sequence[str]], Sequence[float]]


def rerank(
    question: str,
    ranking: Sequence[str],
    texts: Mapping[str, str],
    score: Scorer,
    depth: int = RERANK_DEPTH,
) -> list[str]:
    """A ranking with its first `depth` chunks reordered by score, best first, ties in
    their old order, and the rest after them as they were."""
    head = list(ranking[:depth])
    scores = score(question, [texts[chunk_id] for chunk_id in head])
    order = sorted(range(len(head)), key=lambda place: (-scores[place], place))
    return [head[place] for place in order] + list(ranking[depth:])


# --- the model, on a machine that has it -------------------------------------------


def load_reranker(device: str | None = None, path: str | None = None) -> Any:
    """The cross-encoder, in half precision on a GPU. `path` overrides where it loads from."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(path or RERANKER, device=device, max_length=MAX_TOKENS)
    if model.device.type == "cuda":
        model.half()
    return model


def predict(model: Any, question: str, texts: Sequence[str], batch_size: int = 16) -> list[float]:
    """The cross-encoder's score for the question with each text: its logit, which a
    sigmoid would only squeeze, so that the best chunks do not tie near 1."""
    pairs = [(question, text) for text in texts]
    scores = model.predict(
        pairs, batch_size=batch_size, show_progress_bar=False, activation_fn=_logits
    )
    return [float(score) for score in scores]


def _logits(scores: Any) -> Any:
    return scores
