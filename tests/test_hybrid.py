"""Tests for fusing rankings, reranking their head, and the evaluation over both.

Scores stand in for the cross-encoder, as stored vectors do for the encoders,
so the suite runs without PyTorch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from daleel.retrieve.dense import DOCUMENT, QUESTION, VectorStore
from daleel.retrieve.fusion import rrf
from daleel.retrieve.hybrid import Hybrid, Search, fusions
from daleel.retrieve.rerank import ScoreStore, predict, rerank


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    return path


def ranked(*ids: str) -> Search:
    def search(question: str, k: int) -> list[str]:
        return list(ids[:k])

    return search


def test_fusion_sums_each_ranking_s_reciprocal_rank() -> None:
    # a: 1/61 + 1/63, b: 1/62 + 1/61, c: 1/63 + 1/62
    assert rrf([["a", "b", "c"], ["b", "c", "a"]]) == ["b", "a", "c"]
    assert rrf([["a", "b"], ["c"]], k=0) == ["a", "c", "b"]


def test_fusion_keeps_chunks_of_one_ranking_and_breaks_ties_by_first_appearance() -> None:
    assert rrf([["a", "b"], ["b", "a"]]) == ["a", "b"]
    assert rrf([["a"], ["b"], ["c", "a"]]) == ["a", "b", "c"]
    assert rrf([]) == []


def test_reranking_reorders_the_head_alone_ties_in_their_old_order() -> None:
    texts = {name: f"text {name}" for name in "abcde"}
    asked = []

    def score(question: str, candidates: list[str]) -> list[float]:
        asked.append(candidates)
        return [{"text a": 0.1, "text b": 0.9, "text c": 0.1}[text] for text in candidates]

    assert rerank("q", list("abcde"), texts, score, depth=3) == list("bacde")
    assert asked == [["text a", "text b", "text c"]]


def test_scores_are_kept_through_a_file_and_refused_where_absent(tmp_path: Path) -> None:
    store = ScoreStore()
    store.add("كم الرسوم؟", ["المبلغ 250", "الغياب"], [2.5, -1.0])
    store.meta = {"model": "BAAI/bge-reranker-v2-m3"}
    again = ScoreStore.load(store.save(tmp_path / "rerank" / "scores.json"))
    assert again == store
    assert again.score("كم الرسوم؟", ["الغياب", "المبلغ 250"]) == [-1.0, 2.5]
    assert again.missing("كم الرسوم؟", ["الغياب", "جديد", "جديد"]) == ["جديد"]
    with pytest.raises(KeyError, match="1 chunks have no score"):
        again.score("كم الرسوم؟", ["الغياب", "جديد"])
    with pytest.raises(ValueError, match="1 texts but 2 scores"):
        store.add("q", ["x"], [1.0, 2.0])


def test_the_cross_encoder_reads_the_question_with_each_text_and_gives_logits() -> None:
    class StandIn:
        def predict(self, pairs: list, activation_fn: Callable, **options: object) -> np.ndarray:
            self.pairs, self.activation = pairs, activation_fn
            return np.array([1.5, -2.0], dtype=np.float32)

    model = StandIn()
    assert predict(model, "q", ["x", "y"]) == [1.5, -2.0]
    assert model.pairs == [("q", "x"), ("q", "y")]
    assert model.activation(np.array([9.0]))[0] == 9.0


def test_bm25_is_fused_with_each_encoder_and_with_all_of_them() -> None:
    bm25, first, second = ranked("a"), ranked("b"), ranked("c")
    assert fusions(bm25, {}) == {}
    assert fusions(bm25, {"bge-m3": first}) == {"BM25 + bge-m3": [bm25, first]}
    assert fusions(bm25, {"bge-m3": first, "e5": second}) == {
        "BM25 + bge-m3": [bm25, first],
        "BM25 + e5": [bm25, second],
        "BM25 + bge-m3 + e5": [bm25, first, second],
    }


def test_a_hybrid_fuses_its_retrievers_then_reranks_the_head() -> None:
    texts = {name: name for name in "abcd"}
    bm25, dense = ranked("a", "b", "c"), ranked("c", "d", "a")
    assert Hybrid([bm25]).search("q", 2) == ["a", "b"]
    assert Hybrid([bm25, dense]).search("q", 4) == rrf([list("abc"), list("cda")])

    def score(question: str, candidates: list[str]) -> list[float]:
        return [{"a": 0, "b": 0, "c": 0, "d": 1}[text] for text in candidates]

    # Fused, a and c tie, as do b and d: a, c, b, d. Chunk d is beyond a head of three.
    assert Hybrid([bm25, dense], score, texts, rerank_depth=3).search("q", 4) == list("acbd")
    assert Hybrid([bm25, dense], score, texts, rerank_depth=4).search("q", 4) == list("dacb")


def test_a_hybrid_retrieves_each_ranking_to_its_depth() -> None:
    depths = []

    def search(question: str, k: int) -> list[str]:
        depths.append(k)
        return ["a"]

    Hybrid([search, search], depth=7).search("q", 1)
    assert depths == [7, 7]


def test_a_hybrid_needs_a_retriever_and_the_texts_it_reranks() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        Hybrid([])
    with pytest.raises(ValueError, match="texts"):
        Hybrid([ranked("a")], scorer=lambda question, texts: [0.0] * len(texts))


# --- scripts/eval_hybrid.py ---------------------------------------------------------

GUIDE = "student_guide_2025"
CHUNKS = [
    {"chunk_id": "a", "doc_id": GUIDE, "page": 1, "text": "رسوم الكتب المستعارة"},
    {"chunk_id": "b", "doc_id": GUIDE, "page": 2, "text": "المبلغ 250 ريال عند القبول"},
    {"chunk_id": "c", "doc_id": GUIDE, "page": 3, "text": "الغياب عشرون بالمئة"},
]
QUESTION_TEXT = "كم رسوم التسجيل؟"
GOLD = [
    {
        "qid": "g1",
        "type": "single_clause",
        "lang": "ar",
        "question": QUESTION_TEXT,
        "answerability_status": "supported",
        "split": "dev",
        "source_refs": [
            {"doc_id": GUIDE, "pdf_page": 2, "role": "answer_evidence", "quotes": ["المبلغ 250"]}
        ],
    }
]


def stores(tmp_path: Path, scored: str = "abc") -> list[str]:
    """Vectors that put chunk b nearest the question, and scores that put it first."""
    dense = tmp_path / "dense"
    for name in ("bge-m3", "e5-large-instruct"):
        store = VectorStore(meta={"model": name, "device": "cpu"})
        texts = [chunk["text"] for chunk in CHUNKS]
        store.add(DOCUMENT, texts, [unit(0, 1), unit(1, 0), unit(1, -1)])
        store.add(QUESTION, [QUESTION_TEXT], [unit(1, 0.1)])
        store.save(dense / f"{name}.npz")
    scores = ScoreStore(meta={"model": "stand-in"})
    found = {"a": 1.0, "b": 5.0, "c": 0.0}
    by_id = {chunk["chunk_id"]: chunk["text"] for chunk in CHUNKS}
    scores.add(QUESTION_TEXT, [by_id[name] for name in scored], [found[name] for name in scored])
    scores.save(tmp_path / "rerank.json")
    chunks = write_lines(tmp_path / "chunks.jsonl", CHUNKS)
    gold = write_lines(tmp_path / "gold.jsonl", GOLD)
    return ["--chunks", str(chunks), "--gold", str(gold), "--dense", str(dense)]


def test_the_evaluation_reports_every_retriever_fusion_and_reranking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    argv = stores(tmp_path)
    eval_hybrid = script("eval_hybrid")
    assert eval_hybrid.main([*argv, "--rerank", str(tmp_path / "rerank.json"), "--json"]) == 0
    measures = json.loads(capsys.readouterr().out)
    fused = ["BM25 + bge-m3", "BM25 + e5-large-instruct", "BM25 + bge-m3 + e5-large-instruct"]
    assert list(measures) == [
        "BM25",
        "bge-m3",
        "e5-large-instruct",
        *(f"RRF: {name}" for name in fused),
        *(f"RRF: {name}, reranked" for name in fused),
    ]
    assert measures["BM25"]["mrr@10"] < 1
    assert measures["bge-m3"]["mrr@10"] == 1
    assert all(measures[f"RRF: {name}, reranked"]["mrr@10"] == 1 for name in fused)


def test_the_evaluation_leaves_out_rows_it_has_nothing_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    argv = stores(tmp_path)
    (tmp_path / "dense" / "e5-large-instruct.npz").unlink()
    eval_hybrid = script("eval_hybrid")
    assert eval_hybrid.main([*argv, "--rerank", str(tmp_path / "none.json")]) == 0
    out, err = capsys.readouterr()
    assert [line.split(" | ")[0] for line in out.splitlines()[2:5]] == [
        "| BM25",
        "| bge-m3",
        "| RRF: BM25 + bge-m3",
    ]
    assert "so no rows for e5-large-instruct" in err
    assert "so no reranked rows" in err


def test_the_evaluation_stops_where_a_chunk_to_rerank_has_no_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], script: Callable[[str], ModuleType]
) -> None:
    argv = stores(tmp_path, scored="ab")
    eval_hybrid = script("eval_hybrid")
    assert eval_hybrid.main([*argv, "--rerank", str(tmp_path / "rerank.json")]) == 1
    assert "reranked: 1 chunks have no score" in capsys.readouterr().err


def test_the_scored_pool_covers_every_ranking_the_evaluation_reranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: Callable[[str], ModuleType],
) -> None:
    class StandIn:
        device = "cpu"
        pairs = 0

        def predict(self, pairs: list, **options: object) -> list[float]:
            StandIn.pairs += len(pairs)
            return [float(len(text)) for _, text in pairs]

    argv = stores(tmp_path)
    score_rerank_pool = script("score_rerank_pool")
    monkeypatch.setattr(score_rerank_pool, "load_reranker", lambda device, path: StandIn())
    monkeypatch.setattr(score_rerank_pool, "model_revision", lambda model_id: "abc123")
    scored = [*argv, "--out", str(tmp_path / "rerank"), "--depth", "2"]
    assert score_rerank_pool.main(scored) == 0
    assert StandIn.pairs == 3
    assert score_rerank_pool.main(scored) == 0
    assert StandIn.pairs == 3
    assert "every pair is already scored" in capsys.readouterr().out

    target = tmp_path / "rerank" / "bge-reranker-v2-m3.json"
    assert ScoreStore.load(target).meta["pool"]["rankings"][-1] == (
        "BM25 + bge-m3 + e5-large-instruct"
    )
    eval_hybrid = script("eval_hybrid")
    assert eval_hybrid.main([*argv, "--rerank", str(target), "--rerank-depth", "2"]) == 0
    assert "RRF: BM25 + bge-m3 + e5-large-instruct, reranked" in capsys.readouterr().out
