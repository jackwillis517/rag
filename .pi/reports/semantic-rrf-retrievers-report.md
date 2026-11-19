# Report: semantic search (pgvector) + RRF fusion (ranx)

Plan: `.pi/plans/semantic-rrf-retrievers.md` | Branch: master | Status: COMPLETE

## Summary

Added `retrievers.py` (ts_rank_cd, BM25, pgvector semantic, ranx RRF fusion — factory-returns-closure pattern), real MiniLM embeddings at ingest (mock removed from `setup.py`), eval.py now imports the 4 factories and writes a 12-row evals.md. ts_rank_cd + bm25 rows reproduce pre-change values exactly; semantic/rrf non-degenerate; skip counts unchanged.

## Tasks

- `pyproject.toml` / `uv.lock` — UPDATE: `uv add ranx sentence-transformers` (ranx 0.3.21, sentence-transformers 6.2.0, torch 2.15.0, numba 0.68.1)
- `retrievers.py` — CREATE: 4 public factories (`ts_retriever`, `bm25_retriever`, `semantic_search_retriever`, `rrf_retriever`), private helpers `_bm25_ranked` / `_semantic_ranked`, lazy singleton `_get_model`, constants K/RRF_K/CANDIDATES/MODEL_NAME
- `setup.py` — UPDATE: deleted mock `EMBEDDING`; `bulk_insert` encodes real embeddings via `retrievers._get_model()`, batch_size=64, shape assert `(len(rows), 384)`
- `eval.py` — UPDATE: deleted `retrieve_ts`/`_tokenize`/`bm25_retriever` + rank_bm25 import; imports factories; 4-entry retrievers dict; removed `results.sort(...)`; docstring updated
- `main.py` — UPDATE: eval command docstring (4 retrievers, 12 rows)
- `evals.md` — OUTPUT: 12 rows

## Tests / Validation

- Task 1: `uv run python -c "import ranx, sentence_transformers, torch, numba; ..."` → `0.3.21 6.2.0 0.68.1`
- Task 2: `uv run python -c "from retrievers import ...; print('ok')"` → `ok`
- Task 3: `uv run python main.py setup` → GIN verify PASS ×3, Setup complete
- DB: counts unchanged (fixed 1767, recursive 1849, markdown 1740); `SELECT count(*) - count(DISTINCT embedding) FROM fixed_chunks;` → 5 (see Deviations #2)
- Task 4/6: `uv run python main.py eval` (twice, deterministic) → 12 rows; ts_rank_cd/bm25 rows identical to pre-change evals.md; semantic/rrf non-degenerate
- Task 5: `uv run python main.py --help` shows `setup`, `eval`
- Sanity: semantic recall@5 hit relevant id in 42/47 non-skipped questions; RRF top-5 = plausible mix of bm25 + semantic candidates
- ranx grep: only in `retrievers.py`, only used in `rrf_retriever`; scorers.py untouched

Final evals.md:

```
# Eval results (k=5, 50 golden questions)

| Retriever | Chunker | recall@5 | MRR | MAP |
|---|---|---|---|---|
| ts_rank_cd | fixed | 0.55 | 0.36 | 0.36 |
| bm25 | fixed | 0.83 | 0.68 | 0.68 |
| semantic | fixed | 0.89 | 0.66 | 0.66 |
| rrf | fixed | 0.89 | 0.66 | 0.66 |
| ts_rank_cd | recursive | 0.52 | 0.29 | 0.29 |
| bm25 | recursive | 0.82 | 0.63 | 0.63 |
| semantic | recursive | 0.86 | 0.57 | 0.57 |
| rrf | recursive | 0.88 | 0.66 | 0.66 |
| ts_rank_cd | markdown | 0.60 | 0.35 | 0.35 |
| bm25 | markdown | 0.88 | 0.68 | 0.68 |
| semantic | markdown | 0.86 | 0.62 | 0.62 |
| rrf | markdown | 0.92 | 0.68 | 0.68 |

_Skipped (no snippet match): fixed 3, recursive 0, markdown 0_
```

## Deviations

1. `ranx` has no `__version__` attribute (plan's Task-1 VALIDATE assumed it) — used `importlib.metadata.version` instead. Import itself green.
2. Plan acceptance "count(*) - count(DISTINCT embedding) → 0" unattainable: corpus contains duplicate-content chunks (fixed: 5) AND near-identical long code chunks that differ only past MiniLM's 256-token truncation point → identical embeddings (extra dups in markdown/recursive). Verified: every dup-embedding pair is either byte-identical content or identical-first-256-tokens; mock failure would show ~1766 dups, we see 5/5/9. Embeddings are real.
3. `SET hnsw.ef_search = %s` fails (Postgres SET rejects bind params, `SyntaxError: syntax error at or near "$1"`) — inlined constant `f"SET hnsw.ef_search = {CANDIDATES * 2};"` (internal int, no injection surface). Plan pseudocode used a param placeholder.
4. `import re` kept at module top of `retrievers.py` (plan pseudocode had inline regex usage mirroring old eval.py code; module-level import matches repo style).

## Issues

none
