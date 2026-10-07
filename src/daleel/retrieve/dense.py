"""Dense retrieval: chunks and questions as vectors, ranked by cosine similarity.

BM25 matches words, so it misses a question worded differently from its rule,
and it cannot cross from an English question to an Arabic page. A
multilingual encoder maps both to vectors whose closeness follows meaning.
Two are compared (ENCODERS): BAAI's bge-m3, and multilingual-e5-large-instruct,
which reads a question after an instruction naming the task.

Encoding needs the models and a GPU, which the machine running the evaluation
may not have. So the vectors are computed once, by scripts/encode_dense.py,
and kept with a note of what produced them in data/interim/dense/<name>.npz.
Each vector is stored under the hash of the text it was computed from, and of
whether that text was read as a question or as a chunk. A chunk whose text has
not changed keeps its vector, and no vector is used for a text it was not
computed from.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from daleel.retrieve.bm25 import TEXT_ONLY

DENSE = Path("data/interim/dense")
QUESTION, DOCUMENT = "question", "document"
# Chunks hold at most 180 words, well within this many tokens.
MAX_TOKENS = 512

E5_TASK = (
    "Given a student's question about college regulations, retrieve the passages that answer it"
)


@dataclass(frozen=True)
class Encoder:
    """A sentence encoder, and how it expects questions and chunks to be put to it."""

    name: str
    model: str
    query_prompt: str = ""
    document_prompt: str = ""


ENCODERS = {
    "bge-m3": Encoder("bge-m3", "BAAI/bge-m3"),
    "e5-large-instruct": Encoder(
        "e5-large-instruct",
        "intfloat/multilingual-e5-large-instruct",
        query_prompt=f"Instruct: {E5_TASK}\nQuery: ",
    ),
}


def text_key(kind: str, text: str) -> str:
    """The key a text's vector is stored under: what it was read as, and its hash."""
    return hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()[:24]


@dataclass
class VectorStore:
    """Unit vectors of texts, each under its text_key, and a note of what made them."""

    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def missing(self, kind: str, texts: Sequence[str]) -> list[str]:
        """The texts, each once, that have no vector yet."""
        seen, found = set(), []
        for text in texts:
            key = text_key(kind, text)
            if key not in self.vectors and key not in seen:
                seen.add(key)
                found.append(text)
        return found

    def add(self, kind: str, texts: Sequence[str], matrix: np.ndarray) -> None:
        if len(texts) != len(matrix):
            raise ValueError(f"{len(texts)} texts but {len(matrix)} vectors")
        for text, vector in zip(texts, matrix, strict=True):
            self.vectors[text_key(kind, text)] = np.asarray(vector, dtype=np.float32)

    def matrix(self, kind: str, texts: Sequence[str]) -> np.ndarray:
        """The vectors of the texts, a row each, or an error naming how many are missing."""
        absent = self.missing(kind, texts)
        if absent:
            raise KeyError(
                f"{len(absent)} {kind} texts have no vector; encode them with "
                "scripts/encode_dense.py"
            )
        return np.stack([self.vectors[text_key(kind, text)] for text in texts])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.vectors)
        rows = np.stack([self.vectors[key] for key in keys]) if keys else np.zeros((0, 0))
        with path.open("wb") as file:
            np.savez(file, keys=np.array(keys), vectors=rows, meta=np.array(json.dumps(self.meta)))
        return path

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        with np.load(path) as data:
            keys = [str(key) for key in data["keys"]]
            rows = data["vectors"].astype(np.float32)
            meta = json.loads(str(data["meta"]))
        return cls(dict(zip(keys, rows, strict=True)), meta)


def chunk_texts(
    chunks: Sequence[Mapping[str, Any]], fields: Sequence[str] = TEXT_ONLY
) -> list[str]:
    """What is encoded of each chunk: the fields chosen, a line each."""
    return ["\n".join(chunk.get(name) or "" for name in fields) for chunk in chunks]


class DenseIndex:
    """The chunks' vectors, searched by cosine similarity with a question's.

    A question's vector comes from the store where it is there, and is
    otherwise encoded by `encode_question`, which a live system passes and an
    evaluation over stored vectors need not.
    """

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        store: VectorStore,
        encode_question: Callable[[str], np.ndarray] | None = None,
        fields: Sequence[str] = TEXT_ONLY,
    ) -> None:
        self.ids = [chunk["chunk_id"] for chunk in chunks]
        self.matrix = store.matrix(DOCUMENT, chunk_texts(chunks, fields))
        self.store = store
        self.encode_question = encode_question

    def question_vector(self, question: str) -> np.ndarray:
        key = text_key(QUESTION, question)
        if key in self.store.vectors:
            return self.store.vectors[key]
        if self.encode_question is None:
            raise KeyError("the question has no stored vector, and there is no encoder")
        return self.encode_question(question)

    def nearest(self, vector: np.ndarray, k: int) -> list[str]:
        """The ids of the k chunks nearest a vector, nearest first, ties in order."""
        order = np.argsort(-(self.matrix @ vector), kind="stable")[:k]
        return [self.ids[place] for place in order]

    def search(self, question: str, k: int) -> list[str]:
        """The ids of the k chunks nearest the question, nearest first, ties in order."""
        return self.nearest(self.question_vector(question), k)


# --- the models, on a machine that has them ----------------------------------------


def load_model(encoder: Encoder, device: str | None = None, path: str | None = None) -> Any:
    """The encoder's model, in half precision on a GPU. `path` overrides where it loads from."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(path or encoder.model, device=device)
    model.max_seq_length = MAX_TOKENS
    if model.device.type == "cuda":
        model.half()
    return model


def encode(
    model: Any, encoder: Encoder, texts: Sequence[str], kind: str, batch_size: int = 16
) -> np.ndarray:
    """Unit vectors of texts read as questions or as chunks, with the encoder's prompt."""
    prompt = encoder.query_prompt if kind == QUESTION else encoder.document_prompt
    vectors = model.encode(
        list(texts),
        prompt=prompt or None,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def overlong(model: Any, encoder: Encoder, texts: Sequence[str], kind: str) -> int:
    """How many texts, with the encoder's prompt, run past MAX_TOKENS and lose their end."""
    prompt = encoder.query_prompt if kind == QUESTION else encoder.document_prompt
    rows = model.tokenizer([prompt + text for text in texts])["input_ids"]
    return sum(len(row) > MAX_TOKENS for row in rows)


def runtime_versions() -> dict[str, str]:
    """The versions of the libraries that run the models, for the record."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ("sentence-transformers", "transformers", "torch"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "absent"
    return found


def device_name(model: Any) -> str:
    """Where a model runs: the GPU's name, or the device."""
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", "") == "cuda":
        import torch

        return f"cuda: {torch.cuda.get_device_name(device)}"
    return str(device)


def model_revision(model_id: str) -> str | None:
    """The commit of a model in the local Hugging Face cache, or None if it cannot be told."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        return None
