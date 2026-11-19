# Implementer Memory — project p2

## Project conventions
- Validation = inline `uv run` commands, no pytest.
- Retriever pattern: factory-returns-closure (`retrievers.py`), question -> top-k chunk-id fns.
- Postgres: `postgres://postgres:password@localhost/p2`; pgvector HNSW cosine; `register_vector` in `setup.connect()`.
- Chunk tables: fixed_chunks (1767), recursive_chunks (1849), markdown_chunks (1740).

## Gotchas learned
- Postgres `SET` rejects bind params (`SET hnsw.ef_search = $1` → SyntaxError); inline constants.
- `ranx` has no `__version__`; use `importlib.metadata.version('ranx')`. `Run.from_dict` crashes on empty inner dict → guard. Doc ids must be str.
- MiniLM all-MiniLM-L6-v2: 256-token truncation → near-identical long chunks get identical embeddings; dup-embedding count in corpus ≈ dup-content count (5/5/9 per table).
- ollama 0.6.3: no `__version__` attr either — use `importlib.metadata.version('ollama')`.
- RTX 3080 10GB + qwen3.5:4b @ num_ctx=262144 f16 KV: 13GB resident → 55%/45% CPU/GPU offload → ~40s/call warm. Needs `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` restart (user-side) → ~7.6GB fits GPU.
- Plan VALIDATE blocks can be self-inconsistent (e.g. `diff head -16` vs interleaved new rows) — verify INTENT, log deviation.
- ranx emits SyntaxWarnings (invalid escape sequences) on import — harmless; `python -W ignore` silences.
- torch wheel = full CUDA (torch 2.15.0), per plan default.

## Reports
- semantic+RRF: `.pi/reports/semantic-rrf-retrievers-report.md` (COMPLETE, 2025-09-18-ish session)
- rrf_ce + pipeline: `.pi/reports/rrf-ce-pipeline-report.md` (PARTIAL, 2026-11-06) — tasks 1-3,5 done (evals.md 15 rows, 12 byte-identical); pipeline machinery + markdown_context_chunks (1740 rows, all NULL) built; full generation BLOCKED on user restarting ollama with q8_0 KV flags.
