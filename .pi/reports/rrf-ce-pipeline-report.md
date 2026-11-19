# Report: RRF→cross-encoder rerank row + `pipeline` command (contextual retrieval)

Plan: `.pi/plans/rrf-ce-pipeline.md` | Branch: master | Status: **PARTIAL** (tasks 1–3, 5 COMPLETE; task 4 machinery COMPLETE but full 1740-row context generation BLOCKED on ollama env flags — user must restart ollama; task 6 blocked on 4)

## Summary

5th eval retriever `rrf_ce` (RRF top-10 → ms-marco-MiniLM-L-12-v2 rerank → top-5) implemented; evals.md now 15 rows with the original 12 byte-identical. `pipeline.py` created (contextual-retrieval table, ollama generation, embedding backfill, HNSW, 1-row eval) and all machinery validated, but full generation needs ollama restarted with q8_0 KV cache: current server holds 13GB (55%/45% CPU/GPU split) → ~40s/call → ~19h for 1740 contexts.

## Tasks / Files

- Task 1: `uv add ollama` (0.6.3) → `pyproject.toml`, `uv.lock` (UPDATE)
- Task 2: `retrievers.py` (UPDATE) — `RRF_CE_CANDIDATES=10`, `CROSS_MODEL_NAME`, `_get_cross_encoder` lazy singleton, `_bm25_corpus`, `_bm25_ranked` widened to `body_expr` + `(ids, bodies, bm25)` cache, `_rrf_ranked` extracted (move-verbatim), `rrf_retriever` thin wrapper, `_cross_encoder_rerank`, `rrf_ce_retriever`
- Task 3: `eval.py` (UPDATE) — 5th dict entry `rrf_ce`, docstring 15 rows
- Task 4: `pipeline.py` (CREATE) — `markdown_context_chunks` DDL/populate, ollama context generation (qwen3.5:4b, temp 0, num_predict 128, num_ctx 262144, think=False, batched commits of 32, resume via `WHERE context IS NULL`), embedding backfill over `context\n\ncontent`, HNSW, 1-row `pipeline.md` eval. **Table built: 1740 rows, all context NULL (clean resume state). Generation NOT run — see VRAM.**
- Task 5: `main.py` (UPDATE) — `pipeline` command; eval docstring 5 retrievers / 15 rows

## Validation outputs

- Task 1: `uv run python -c "import ollama"` → `0.6.3` (note: no `ollama.__version__` attr; `importlib.metadata.version('ollama')` — same as ranx gotcha)
- Task 2: imports ok; smoke `rrf_ce_retriever(conn, 'markdown_chunks', {})('How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?')` → `[1311, 1310, 1312, 1309, 1263]` (5 int ids); rerank seam: logits `[-11.19, +10.00]`, order `[2, 1]` with meaningful question
- Task 3: `uv run python main.py eval` → 15 rows; `grep -v '^| rrf_ce ' evals.md | diff /tmp/evals.before.md -` → EMPTY (12 rows byte-identical); skipped line unchanged (`fixed 3, recursive 0, markdown 0`); `grep -c '^| rrf_ce' evals.md` → 3
- Task 4: `ensure_table` → 1740 rows; live `_one_context` scratch tests (see VRAM below) — contexts sane, non-empty, `think=False` working; sandbox (fake contexts, 20 rows) validated `backfill_embeddings` + HNSW + `BODY_EXPR` SQL + `rrf_ce_retriever(body_expr=BODY_EXPR)` + evaluate end-to-end; `pipeline.md` write path = eval.py:86-99 pattern (validated by task 3)
- Task 5: `uv run python main.py --help` → setup / eval / pipeline
- L1: `import retrievers, pipeline, eval, main` ok

## evals.md (15 rows, current)

```
| ts_rank_cd | fixed | 0.55 | 0.36 | 0.36 |
| bm25 | fixed | 0.83 | 0.68 | 0.68 |
| semantic | fixed | 0.89 | 0.66 | 0.66 |
| rrf | fixed | 0.89 | 0.66 | 0.66 |
| rrf_ce | fixed | 0.91 | 0.83 | 0.83 |
| ts_rank_cd | recursive | 0.52 | 0.29 | 0.29 |
| bm25 | recursive | 0.82 | 0.63 | 0.63 |
| semantic | recursive | 0.86 | 0.57 | 0.57 |
| rrf | recursive | 0.88 | 0.66 | 0.66 |
| rrf_ce | recursive | 0.94 | 0.81 | 0.81 |
| ts_rank_cd | markdown | 0.60 | 0.35 | 0.35 |
| bm25 | markdown | 0.88 | 0.68 | 0.68 |
| semantic | markdown | 0.86 | 0.62 | 0.62 |
| rrf | markdown | 0.92 | 0.68 | 0.68 |
| rrf_ce | markdown | 0.92 | 0.81 | 0.81 |
```

rrf_ce ≥ rrf on recall@5/MRR/MAP in all 3 chunkers (soft expectation met).

## VRAM / ollama observations (blocking issue)

- GPU: RTX 3080 10GB, ~1.6GB in use by desktop/other → ~8.25GB free.
- `ollama ps` during generation: **qwen3.5:4b, SIZE 13 GB, 55%/45% CPU/GPU, CONTEXT 262144** — model 3.39GB (4.7B params incl. vision projector, Q4_K_M) + f16 KV ≈ 8.4GB/slot at num_ctx=262144 > VRAM → ollama CPU-offloads ~45% → decode ~3-4 tok/s.
- No `OLLAMA_*` env flags set on the systemd unit (`systemctl show ollama -p Environment` = only PATH).
- Measured: cold prefill cloud_run doc (~74k tok): 87–117s/call; cold prefill letsgo doc (~150k tok): 232s/call; warm (prefix-cached, CPU-offloaded): 37–65s/call. At ~40s/call × 1740 ≈ **~19h** — unacceptable.
- Generation itself is CORRECT (no OOM, num_ctx honored — model sees chunk after the 150k-token document; contexts reference real doc content, e.g. "Chapter 14 / Mailtrap email check"; non-empty → think=False working).
- **User must restart ollama with** (systemd override: `sudo systemctl edit ollama` → `[Service]` → `Environment=...`, then `sudo systemctl restart ollama`):
  ```
  OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
  ```
  q8_0 KV ≈ 4.2GB → 3.39 + 4.2 ≈ 7.6GB → fully on GPU; warm decode 50–100 tok/s → warm call ≈ a few s; full run est. minutes-to-low-hours (one-time; cached forever after). num_ctx stays 262144 — NO document truncation (plan requirement).
- After user restarts with flags: `uv run python main.py pipeline` resumes (all 1740 context NULL → generates all, backfills, evaluates, writes pipeline.md).

## Deviations

1. `CONTEXT_WORKERS = 4 → 1` in pipeline.py — per user VRAM instruction (memory-bound; ollama server effectively 1 slot anyway).
2. Plan's task-3 VALIDATE `diff <(head -16 ...) <(head -16 evals.md)` cannot be empty: rrf_ce rows interleave per-table block (plan's own dict insertion order = row order), so lines shift. Verified the INTENT instead: 12 non-rrf_ce rows byte-identical to pre-change file (`grep -v '^| rrf_ce ' | diff` → empty). Plan instruction self-inconsistent; no code impact.
3. Plan's rerank smoke test used degenerate question `'q'` → model returns `[1, 2]` (both passages irrelevant to "q"; logits near-tie noise). With meaningful question (`'What is the population of Berlin?'`) → `[2, 1]` as expected. Not a code defect.
4. `ollama.__version__` doesn't exist in 0.6.3 (plan's task-1 VALIDATE as written fails) — used `importlib.metadata.version('ollama')` → 0.6.3.

## Issues

- BLOCKED: full pipeline run (task 4 generation, task 6) awaits ollama restart with the 3 env flags above. Everything else green. Table in clean resume state (1740 rows, context/embedding NULL). If q8_0 KV still OOMs at 262144, options: (a) check other GPU processes free VRAM, (b) CPU-only generation (hours), (c) plan change to truncate docs — explicitly rejected by plan.
- `pipeline.md` not yet written (needs full generation + backfill + eval). Row shape/eval path validated via sandbox.
