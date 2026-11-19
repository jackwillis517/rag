"""Eval harness: 5 retrievers (ts_rank_cd, BM25, semantic, RRF fusion,
RRF + Cross Encoder Rerank) across the three chunker-variant tables, scored
on the golden set.

Relevance = snippet containment in chunk content (60-char prefix fallback).
Questions whose snippet matches no chunk are skipped and counted per chunker.
Writes evals.md (15 rows) and prints it.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

from retrievers import (
    bm25_retriever,
    rrf_ce_retriever,
    rrf_retriever,
    semantic_search_retriever,
    ts_retriever,
)
from scorers import average_precision, mrr, recall_at_k
from setup import TABLES, connect

GOLDEN_PATH = Path("golden_set.JSON")
EVALS_PATH = Path("evals.md")
K = 5


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def relevant_ids(rows: list[tuple[int, str]], snippet: str) -> set[int]:
    """Ids of chunks whose normalized content contains the normalized snippet."""
    n_snip = _norm(snippet)
    ids = {id for id, content in rows if n_snip in _norm(content)}
    if not ids:
        prefix = n_snip[:60]
        ids = {id for id, content in rows if prefix in _norm(content)}
    return ids


def evaluate(
    retriever: Callable[[str], list[int]],
    golden: list[dict],
    relevant: list[set[int]],
    k: int = K,
) -> dict:
    """Mean recall@k / MRR / MAP over non-skipped questions."""
    recalls, mrrs, aps = [], [], []
    for q, rel in zip(golden, relevant):
        if not rel:
            continue  # skipped question: excluded from the denominator
        retrieved = retriever(q["question"])
        recalls.append(recall_at_k(retrieved, rel, k))
        mrrs.append(mrr(retrieved, rel, k))
        aps.append(average_precision(retrieved, rel, k))
    n = len(recalls)
    return {
        "recall": sum(recalls) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
        "map": sum(aps) / n if n else 0.0,
        "n": n,
    }


def run() -> None:
    golden = load_golden()
    cache: dict = {}
    results: list[tuple[str, str, dict]] = []  # (retriever, chunker, metrics)
    skipped: dict[str, int] = {}

    with connect() as conn:
        for table in TABLES:
            chunker = table.removesuffix("_chunks")
            with conn.cursor() as cur:
                cur.execute(f"SELECT id, content FROM {table}")
                rows = cur.fetchall()
            relevant = [relevant_ids(rows, q["ground_truth_snippet"]) for q in golden]
            skipped[chunker] = sum(1 for r in relevant if not r)

            retrievers = {
                "ts_rank_cd": ts_retriever(conn, table),
                "bm25": bm25_retriever(conn, table, cache),
                "semantic": semantic_search_retriever(conn, table),
                "rrf": rrf_retriever(conn, table, cache),
                "rrf_ce": rrf_ce_retriever(conn, table, cache),
            }
            for name, fn in retrievers.items():
                results.append((name, chunker, evaluate(fn, golden, relevant)))

    lines = [
        f"# Eval results (k={K}, {len(golden)} golden questions)",
        "",
        "| Retriever | Chunker | recall@5 | MRR | MAP |",
        "|---|---|---|---|---|",
    ]
    for name, chunker, m in results:
        lines.append(f"| {name} | {chunker} | {m['recall']:.2f} | {m['mrr']:.2f} | {m['map']:.2f} |")
    lines.append("")
    lines.append("_Skipped (no snippet match): " + ", ".join(f"{c} {n}" for c, n in skipped.items()) + "_")
    out = "\n".join(lines) + "\n"

    from variants import load_variants_section  # lazy: variants.py imports eval (circular)

    preserved = load_variants_section()
    if preserved:
        out = out.rstrip("\n") + "\n\n" + preserved
    EVALS_PATH.write_text(out)
    print(out)
    print(f"Wrote {EVALS_PATH}")


if __name__ == "__main__":
    run()