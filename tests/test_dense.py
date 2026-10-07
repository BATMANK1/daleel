"""Tests for dense retrieval over stored vectors, and the script that stores them.

The models are not loaded here: a stand-in with the same calls takes their
place, so the suite runs without PyTorch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from daleel.retrieve.dense import (
    DOCUMENT,
    ENCODERS,
    MAX_TOKENS,
    QUESTION,
    DenseIndex,
    Encoder,
    VectorStore,
    encode,
    overlong,
    text_key,
)

E5 = ENCODERS["e5-large-instruct"]


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text}


CHUNKS = [chunk("a", "الغياب"), chunk("b", "الرسوم"), chunk("c", "الغياب أيضا")]


def stored(questions: dict[str, np.ndarray]) -> VectorStore:
    store = VectorStore()
    store.add(DOCUMENT, ["الغياب", "الرسوم", "الغياب أيضا"], [unit(1, 0), unit(0, 1), unit(1, 0)])
    store.add(QUESTION, list(questions), list(questions.values()))
    return store


class StandIn:
    """Answers the calls the scripts make of a sentence encoder, and records them."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[tuple[list[str], str | None]] = []

    def encode(self, texts: list[str], prompt: str | None = None, **options: object) -> np.ndarray:
        assert options["normalize_embeddings"] is True
        self.calls.append((texts, prompt))
        return np.array([unit(len(text), 1) for text in texts], dtype=np.float64)

    def tokenizer(self, texts: list[str]) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


def test_a_text_is_keyed_apart_as_a_question_and_as_a_chunk() -> None:
    assert text_key(QUESTION, "الغياب") != text_key(DOCUMENT, "الغياب")
    assert text_key(QUESTION, "الغياب") == text_key(QUESTION, "الغياب")


def test_the_store_lists_each_text_without_a_vector_once() -> None:
    store = stored({})
    assert store.missing(DOCUMENT, ["الرسوم", "جديد", "جديد", "آخر"]) == ["جديد", "آخر"]
    assert store.missing(QUESTION, ["الرسوم"]) == ["الرسوم"]


def test_the_store_keeps_its_vectors_and_note_through_a_file(tmp_path: Path) -> None:
    store = stored({"كم الغياب؟": unit(1, 1)})
    store.meta = {"model": "BAAI/bge-m3", "device": "cpu"}
    again = VectorStore.load(store.save(tmp_path / "dense" / "bge-m3.npz"))
    assert again.meta == store.meta
    assert again.vectors.keys() == store.vectors.keys()
    for key, vector in store.vectors.items():
        assert again.vectors[key].dtype == np.float32
        np.testing.assert_array_equal(again.vectors[key], vector)


def test_vectors_are_refused_for_texts_they_were_not_computed_from() -> None:
    store = stored({})
    with pytest.raises(KeyError, match="2 document texts have no vector"):
        store.matrix(DOCUMENT, ["الغياب", "الغياب.", "جديد"])
    with pytest.raises(ValueError, match="2 texts but 1 vectors"):
        store.add(DOCUMENT, ["x", "y"], [unit(1, 0)])


def test_chunks_rank_by_closeness_to_the_question_ties_in_chunk_order() -> None:
    index = DenseIndex(CHUNKS, stored({"كم الغياب؟": unit(1, 0.1)}))
    assert index.search("كم الغياب؟", 3) == ["a", "c", "b"]
    assert index.nearest(unit(0, 1), 1) == ["b"]


def test_a_question_without_a_vector_is_encoded_only_by_an_encoder_given() -> None:
    with pytest.raises(KeyError, match="no stored vector"):
        DenseIndex(CHUNKS, stored({})).search("كم الرسوم؟", 1)
    live = DenseIndex(CHUNKS, stored({}), encode_question=lambda question: unit(0, 1))
    assert live.search("كم الرسوم؟", 1) == ["b"]


def test_a_chunk_whose_text_changed_has_no_vector() -> None:
    with pytest.raises(KeyError, match="1 document texts"):
        DenseIndex([*CHUNKS[:2], chunk("c", "الغياب أيضاً")], stored({}))


def test_only_questions_get_the_instruction_and_vectors_come_out_as_float32() -> None:
    model = StandIn()
    questions = encode(model, E5, ["كم الغياب؟"], QUESTION)
    encode(model, E5, ["الغياب"], DOCUMENT)
    assert model.calls == [(["كم الغياب؟"], E5.query_prompt), (["الغياب"], None)]
    assert E5.query_prompt.startswith("Instruct: ") and E5.query_prompt.endswith("\nQuery: ")
    assert questions.dtype == np.float32 and questions.shape == (1, 2)


def test_texts_longer_than_the_model_reads_are_counted_with_their_prompt() -> None:
    model = StandIn()
    words = "كلمة " * MAX_TOKENS
    prompted = Encoder("x", "x", query_prompt="Query: one ")
    assert overlong(model, prompted, [words, words + "كلمة"], DOCUMENT) == 1
    assert overlong(model, prompted, [words], QUESTION) == 1


# --- scripts/encode_dense.py --------------------------------------------------------


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    return path


def test_the_encoding_script_encodes_only_what_has_no_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    encode_dense = script("encode_dense")
    models: list[StandIn] = []

    def load(encoder: Encoder, device: str | None = None, path: str | None = None) -> StandIn:
        models.append(StandIn())
        return models[-1]

    monkeypatch.setattr(encode_dense, "load_model", load)
    monkeypatch.setattr(encode_dense, "model_revision", lambda model_id: "abc123")
    chunks = write_lines(tmp_path / "chunks.jsonl", CHUNKS)
    gold = write_lines(tmp_path / "gold.jsonl", [{"question": "كم الغياب؟"}])
    out = tmp_path / "dense"
    argv = ["--encoders", "e5-large-instruct", "--chunks", str(chunks), "--gold", str(gold)]
    argv += ["--out", str(out)]

    assert encode_dense.main(argv) == 0
    store = VectorStore.load(out / "e5-large-instruct.npz")
    assert store.meta["revision"] == "abc123"
    assert store.meta["encoded"] == {DOCUMENT: 3, QUESTION: 1}
    assert models[0].calls[1] == (["كم الغياب؟"], E5.query_prompt)

    assert encode_dense.main(argv) == 0
    assert len(models) == 1
    assert "every vector is already in" in capsys.readouterr().out

    write_lines(chunks, [*CHUNKS[:2], chunk("c", "الحضور")])
    assert encode_dense.main(argv) == 0
    assert models[1].calls == [(["الحضور"], None)]
    store = VectorStore.load(out / "e5-large-instruct.npz")
    assert len(store.vectors) == 4
    assert store.missing(DOCUMENT, ["الغياب أيضا"]) == ["الغياب أيضا"]
    assert store.meta["encoded"] == {DOCUMENT: 1, QUESTION: 0}
