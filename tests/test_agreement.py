"""Tests for agreement between two sets of labels: a confusion table and Cohen's kappa."""

from __future__ import annotations

import pytest

from daleel.eval.agreement import cohen_kappa, confusion

LABELS = ("correct", "partial", "incorrect", "hallucinated")


def test_the_table_counts_each_pair_first_labeller_by_row() -> None:
    human = ["correct", "correct", "partial", "hallucinated", "incorrect"]
    judge = ["correct", "partial", "partial", "incorrect", "incorrect"]
    assert confusion(human, judge, LABELS) == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ]


def test_kappa_is_agreement_beyond_what_chance_would_give() -> None:
    # 50 answers: both say yes on 20, both no on 15, and they differ on 15.
    # Observed agreement 0.70, chance 0.50, so kappa (0.70 - 0.50) / (1 - 0.50).
    first = ["yes"] * 25 + ["no"] * 25
    second = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(first, second, ("yes", "no")) == pytest.approx(0.4)


def test_kappa_is_one_for_full_agreement_and_negative_for_systematic_disagreement() -> None:
    human = ["correct", "partial", "incorrect", "hallucinated"]
    assert cohen_kappa(human, human, LABELS) == pytest.approx(1.0)
    assert cohen_kappa(["correct", "partial"], ["partial", "correct"], LABELS) == pytest.approx(-1)


def test_labels_that_cannot_be_compared_are_refused() -> None:
    with pytest.raises(ValueError, match="2 labels against 1"):
        confusion(["correct", "partial"], ["correct"], LABELS)
    with pytest.raises(ValueError, match="no labels"):
        cohen_kappa([], [], LABELS)
    with pytest.raises(ValueError, match="unknown labels: right"):
        confusion(["right"], ["correct"], LABELS)
    # Both giving one and the same label throughout leaves nothing beyond chance.
    with pytest.raises(ValueError, match="undefined"):
        cohen_kappa(["correct"] * 3, ["correct"] * 3, LABELS)
