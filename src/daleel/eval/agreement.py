"""Agreement metrics for two sets of categorical labels."""

from __future__ import annotations

from collections.abc import Sequence


def confusion(
    first: Sequence[str], second: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Count paired labels, with `first` as rows and `second` as columns."""
    if len(first) != len(second):
        raise ValueError(f"{len(first)} labels against {len(second)}")
    if not first or not labels:
        raise ValueError("no labels to compare")

    indices = {label: index for index, label in enumerate(labels)}
    unknown = list(
        dict.fromkeys(
            label for sequence in (first, second) for label in sequence if label not in indices
        )
    )
    if unknown:
        raise ValueError(f"unknown labels: {', '.join(unknown)}")

    table = [[0] * len(labels) for _ in labels]
    for left, right in zip(first, second, strict=True):
        table[indices[left]][indices[right]] += 1
    return table


def cohen_kappa(first: Sequence[str], second: Sequence[str], labels: Sequence[str]) -> float:
    """Compute Cohen's kappa: agreement beyond chance between two labellers."""
    table = confusion(first, second, labels)
    total = len(first)
    row_counts = [sum(row) for row in table]
    column_counts = [sum(column) for column in zip(*table, strict=True)]
    diagonal = sum(table[i][i] for i in range(len(table)))

    # Compute with integer counts to avoid rounding a chance score near 1.
    chance_count = sum(a * b for a, b in zip(row_counts, column_counts, strict=True))
    denominator = total * total - chance_count
    if denominator == 0:
        raise ValueError("Cohen's kappa is undefined when chance agreement is 1")
    return (diagonal * total - chance_count) / denominator
