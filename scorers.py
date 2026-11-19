"""Ranking metrics over retrieved id sequences vs a set of relevant ids."""

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """Fraction of relevant ids present in the top k retrieved."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """Reciprocal rank of the first relevant id in the top k."""
    for i, id in enumerate(retrieved[:k]):
        if id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved: Sequence[int], relevant: set[int], k: int = 5) -> float:
    """Mean of precision@i over relevant positions in the top k, normalized by |relevant|."""
    if not relevant:
        return 0.0
    hits = 0
    ap = 0.0
    for i, id in enumerate(retrieved[:k]):
        if id in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)