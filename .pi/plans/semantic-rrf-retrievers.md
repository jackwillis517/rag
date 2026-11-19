# Plan: semantic search (pgvector) + RRF fusion (ranx), retrievers extracted to `retrievers.py`

Type: enhance + refactor | Complexity: M | Systems: Postgres (pgvector `<=>`/HNSW), Python 3.12/uv
Builds on: PLAN.md restructure (3 chunker tables, golden-set eval, Typer CLI)
Deps to add: `ranx` (0.3.21), `sentence-transformers` (6.x, pulls `torch>=2.2`)

## Goal

As dev, I want real semantic retrieval (pgvector cosine) and reciprocal-rank-fusion (semantic + BM25
via ranx) evaluated across the 3 chunker tables with the same golden set and metrics, all retrievers
living in one `retrievers.py`, so that chunking strategy is comparable across keyword AND semantic
retrieval. evals.md grows 6 → 12 rows.

## Out of scope

- Not: changing relevance computation (snippet containment + 60-char prefix fallback, unchanged — eval.py:24-33)
- Not: changing K (=5), golden set, skip logic, evals.md table columns
- Not: replacing scorers.py with ranx evaluate (ranx used ONLY for RRF fusion, per request)
- Not: pytest (project convention: inline `uv run` validation)
- Not: ts_rank_cd / BM25 behavior changes (moved verbatim)

## Verified facts (grounded)

- `setup.py:26` `EMBEDDING = np.full(384, 0.01, dtype=np.float32)` mock — all rows identical → degenerate cosine. Must be replaced by real embeddings at ingest.
- `setup.py:80-88` HNSW index `USING hnsw (embedding vector_cosine_ops)` → supports pgvector cosine distance operator `<=>`. `setup.py:connect()` does `register_vector(conn)`.
- pgvector-python psycopg: `register_vector` registers dumpers for `pgvector.Vector` AND `numpy.ndarray` (pgvector/psycopg/vector.py `register_vector_info`). `SentenceTransformer.encode()` returns `np.ndarray` → passes as `%s` param directly. Plain `list[float]` does NOT have a registered dumper — wrap in `np.asarray(...)` or `Vector(...)`.
- all-MiniLM-L6-v2 = 384-dim (HF card: "maps sentences & paragraphs to a 384 dimensional dense vector space") → matches `VECTOR(384)` column.
- ranx 0.3.21 API (source: ranx/fusion/rrf.py, ranx/meta/fuse.py, ranx/data_structures/run.py):
  - `ranx.fuse(runs: List[Run], norm: str = "min-max", method: str = "wsum", params: dict = None) -> Run`; `method="rrf"` dispatches to `ranx.fusion.rrf(runs, k=60, name="rrf")` — RRF constant **k=60** default. Pass `params={"k": 60}` explicitly.
  - `fuse` asserts `len(runs) > 1` and that all runs share identical query-id sets.
  - `Run.from_dict(d: Dict[str, Dict[str, float]])` — per-query doc entries are sorted **score DESC** (data_structures/common.py `sort_dict_by_value` argsorts `[-v]`). RRF rank = position in that order. Scores only determine order, not magnitude.
  - doc ids must be `str` (numba `unicode_type`; `to_unicode(doc_id) = doc_id + ""`). Our chunk ids are `int` (SERIAL) → `str(id)` in, `int(...)` out.
  - `Run.from_dict` crashes on empty inner dict (`max()` over empty) → guard.
  - `fused.to_dict()[q_id]` → `{doc_id: fused_score}` in desc-score insertion order; still tie-break sort ourselves for determinism.
  - `ranx.fuse(..., norm=None)` skips normalization (RRF ignores score magnitude anyway) — use `norm=None`.
  - ranx deps: numpy, numba>=0.54.1, pandas, tabulate, tqdm, scipy>=1.8, ir-datasets, rich, orjson, lz4, cbor2, seaborn, fastparquet — **no torch**. numba 0.68.1 allows numpy<2.6 → project numpy 2.5.4 OK.
- `sentence-transformers` 6.2.0 requires `torch>=2.2`, `transformers>=5.0` — heavy (linux torch wheel pulls CUDA libs, multi-GB unless CPU index pinned). First `SentenceTransformer(MODEL_NAME)` downloads ~90MB from HF hub.
- eval.py current shape: `evaluate(retriever: Callable[[str], list[int]], golden, relevant, k=K)` (eval.py:62) is already retriever-agnostic — no signature change needed. `run()` sorts results `(r[0] != "ts_rank_cd",)` (eval.py:100).

## File layout

- `retrievers.py` — NEW: 4 retriever factories + shared model loader + constants
- `setup.py` — UPDATE: real embeddings at ingest (drop mock `EMBEDDING`)
- `eval.py` — UPDATE: delete `retrieve_ts`/`_tokenize`/`bm25_retriever`, import from `retrievers.py`, 4-retriever loop, row order
- `main.py` — UPDATE: docstring text only
- `evals.md` — OUTPUT: 12 rows
- `pyproject.toml` / `uv.lock` — UPDATE via `uv add`

## retrievers.py

```python
"""Retrievers: Postgres FTS, BM25, pgvector semantic, ranx RRF fusion.

All public functions are factories returning question -> top-k chunk-id
closures, mirroring the original bm25_retriever pattern (eval.py:45).
"""

K = 5                    # same as eval.K; closure default
RRF_K = 60               # ranx RRF constant (Cormack et al. default)
CANDIDATES = 50          # per-retriever depth fed INTO fusion (>= K)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, matches VECTOR(384)

_MODEL = None
def _get_model():
    # lazy singleton: module import stays cheap; setup.py reuses this for ingestion
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def _tokenize(s: str) -> list[str]
    # re.findall(r"[a-z0-9]+", s.lower()) — MOVED verbatim from eval.py:41-42

def ts_retriever(conn, table: str) -> Callable[[str], list[int]]
    # factory; closure(question, k=K):
    #   terms = " | ".join(re.findall(r"[a-z0-9]+", question.lower()))
    #   SELECT id FROM {table}
    #     WHERE tsv @@ to_tsquery('english', %s)
    #     ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC
    #     LIMIT %s
    # MOVED verbatim from eval.py:35-52 (was retrieve_ts)

def _bm25_ranked(conn, table: str, cache: dict) -> Callable[[str, int], list[tuple[int, float]]]
    # per-table cache {(ids, BM25Okapi)} — PRESERVED pattern from eval.py:45-60
    # closure(question, depth) -> [(id, bm25_score), ...] best-first, len == depth
    #   scores = bm25.get_scores(_tokenize(question))
    #   order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:depth]

def bm25_retriever(conn, table: str, cache: dict) -> Callable[[str], list[int]]
    # ranked = _bm25_ranked(conn, table, cache)
    # closure(question, k=K) -> [id for id, _ in ranked(question, k)]

def _semantic_ranked(conn, table: str, model) -> Callable[[str, int], list[tuple[int, float]]]
    # closure(question, depth) -> [(id, cosine_similarity), ...] best-first
    #   qvec = model.encode(question)                       # np.ndarray shape (384,)
    #   with conn.cursor() as cur:
    #     cur.execute("SET hnsw.ef_search = %s;", (CANDIDATES * 2,))   # >= depth, default 40 < 50
    #     cur.execute(
    #       f"""SELECT id, 1 - (embedding <=> %s) AS sim FROM {table}
    #           ORDER BY embedding <=> %s, id ASC LIMIT %s""",
    #       (qvec, qvec, depth))
    #   return [(row[0], float(row[1])) for row in cur.fetchall()]
    # NOTE: qvec passed twice (can't ORDER BY alias with the param inline); ndarray dumper
    # registered by setup.connect()'s register_vector. `id ASC` = deterministic ties.

def semantic_search_retriever(conn, table: str) -> Callable[[str], list[int]]
    # ranked = _semantic_ranked(conn, table, _get_model())
    # closure(question, k=K) -> [id for id, _ in ranked(question, k)]

def rrf_retriever(conn, table: str, cache: dict) -> Callable[[str], list[int]]
    # model = _get_model(); sem = _semantic_ranked(...); bm = _bm25_ranked(...)
    # closure(question, k=K):
    #   sem_list = sem(question, CANDIDATES); bm_list = bm(question, CANDIDATES)
    #   if not sem_list: return [id for id, _ in bm_list[:k]]      # ranx needs non-empty runs
    #   if not bm_list:  return [id for id, _ in sem_list[:k]]
    #   run_sem = Run.from_dict({"q": {str(id): sim for id, sim in sem_list}})
    #   run_bm  = Run.from_dict({"q": {str(id): score for id, score in bm_list}})
    #   fused = fuse(runs=[run_sem, run_bm], norm=None, method="rrf", params={"k": RRF_K})
    #   ranked = sorted(fused.to_dict()["q"].items(), key=lambda kv: (-kv[1], int(kv[0])))
    #   return [int(doc_id) for doc_id, _ in ranked[:k]]
    # from ranx import Run, fuse  (module top)
```

## setup.py changes

```python
# DELETE: EMBEDDING = np.full(384, 0.01, dtype=np.float32)   (setup.py:26)

def bulk_insert(conn, table, title, rows) -> None:
    # rows unchanged: (chunk_index, content, heading_path)
    # model = _get_model()                                    # from retrievers import
    # embeddings = model.encode([content for _, content, _ in rows], batch_size=64)
    # assert embeddings.shape == (len(rows), 384)             # guard dim == VECTOR(384)
    # INSERT ... VALUES (%s, %s, %s, %s, %s) with embedding=embeddings[i] per row
    # (ndarray dumper registered by register_vector in connect(); keep executemany+commit)
```

Import: `from retrievers import _get_model` — no circular import (retrievers.py imports nothing from setup.py). numpy import stays (assert not needed if shape checked; keep for embedding handling).

## eval.py changes

```python
# DELETE: retrieve_ts (35-52), _tokenize (41-42), bm25_retriever (45-60), rank_bm25 import
# ADD:    from retrievers import bm25_retriever, rrf_retriever, semantic_search_retriever, ts_retriever
# KEEP:  load_golden, _norm, relevant_ids, evaluate, K=5, GOLDEN_PATH, EVALS_PATH — unchanged

def run() -> None:
    # ... relevance/skip per table unchanged ...
    retrievers = {   # row order in evals.md = this order; REMOVE old results.sort (eval.py:100)
        "ts_rank_cd": ts_retriever(conn, table),
        "bm25": bm25_retriever(conn, table, cache),
        "semantic": semantic_search_retriever(conn, table),
        "rrf": rrf_retriever(conn, table, cache),
    }
    # evaluate(fn, golden, relevant) unchanged — closures bind k=K
```

12 rows: 4 retrievers × 3 chunkers, table order as `TABLES` (fixed, recursive, markdown).

## Tasks (dependency order)

### 1. UPDATE `pyproject.toml`

- DO: `uv add ranx sentence-transformers`
- GOTCHA: linux `torch` wheel pulls CUDA libs (~2-3GB). Optional slim: `uv add torch --index-url https://download.pytorch.org/whl/cpu` first, then the rest. Keep default unless disk-poor.
- GOTCHA: numba (ranx dep) pins numpy <2.6 — project pins numpy>=2.5.4, resolves fine (numba 0.68.1).
- VALIDATE: `uv run python -c "import ranx, sentence_transformers, torch, numba; print(ranx.__version__, sentence_transformers.__version__)"`

### 2. CREATE `retrievers.py` (depends on 1)

- DO: per signatures above. `_get_model` lazy; `_bm25_ranked`/`_semantic_ranked` private score-returning helpers; 4 public factories.
- PATTERN: factory-returns-closure — eval.py:45-60 (`bm25_retriever`); ts query — eval.py:35-52.
- GOTCHA: pgvector `<=>` = cosine DISTANCE (0 = identical) → ORDER BY ascending; sim = 1 - dist.
- GOTCHA: ranx doc ids must be str; empty candidate list crashes `Run.from_dict` → guards.
- VALIDATE: `uv run python -c "from retrievers import ts_retriever, bm25_retriever, semantic_search_retriever, rrf_retriever; print('ok')"`

### 3. UPDATE `setup.py` (depends on 2)

- DO: delete mock `EMBEDDING`; encode real embeddings in `bulk_insert` (content only, not heading_path — matches tsv weighting where heading is boost, not body); import `_get_model`.
- GOTCHA: first run downloads model (~90MB) — needs network once; cached under ~/.cache.
- GOTCHA: re-running `setup` re-drops/re-creates tables → re-ingests → embeddings regenerate. Already idempotent by design; required for real embeddings.
- VALIDATE: `uv run python main.py setup` then
  `psql postgres://postgres:password@localhost/p2 -c "SELECT count(*) - count(DISTINCT embedding) FROM fixed_chunks;"` → 0 (all distinct), and `-c "SELECT count(*) FROM fixed_chunks;"` matches prior run count.

### 4. UPDATE `eval.py` (depends on 2; tables from 3)

- DO: delete moved retrievers, import factories, 4-entry retrievers dict, remove `results.sort(...)`, update module docstring.
- GOTCHA: `results.sort(key=lambda r: (r[0] != "ts_rank_cd",))` (eval.py:100) would alphabetize non-ts rows — delete it; dict insertion order is the intended row order.
- GOTCHA: keep `cache: dict` shared across bm25/rrf factories per table so BM25 index built once (rrf reuses same cache entry).
- VALIDATE: `uv run python main.py eval` → evals.md has 12 rows, header/footnote format unchanged.

### 5. UPDATE `main.py` (depends on 4)

- DO: docstrings mention 4 retrievers/12 rows. Commands unchanged.
- VALIDATE: `uv run python main.py --help` shows `setup`, `eval`.

### 6. Full-pipeline validation (depends on all)

- DO: run setup → eval fresh; diff 12-row evals.md.
- VALIDATE: `uv run python main.py setup && uv run python main.py eval`
- Expect: ts_rank_cd + bm25 rows identical to current evals.md (same code path, same relevance); semantic/rrf rows non-degenerate (not all 0.00); skipped counts unchanged (fixed 3, recursive 0, markdown 0).
- Sanity: `uv run python -c "` spot-check: for 2-3 golden questions, semantic top-5 contains the snippet-containment relevant id more often than chance; RRF top-5 = plausible mix of bm25 + semantic candidates.

## Acceptance

- [ ] `retrievers.py` holds all 4 retrievers; eval.py imports them; no retriever SQL/logic remains in eval.py
- [ ] DB embeddings: `count(DISTINCT embedding)` == `count(*)` per table (real, not mock)
- [ ] evals.md: 12 rows, stable order (ts_rank_cd, bm25, semantic, rrf) × (fixed, recursive, markdown)
- [ ] Same golden set, same relevance fn, same skip counts as before the change
- [ ] ranx used ONLY in `rrf_retriever` (fusion); scorers.py untouched
- [ ] ts_rank_cd + bm25 metric rows reproduce pre-change values

## Open / assumed

- ASSUMED — sentence-transformers + all-MiniLM-L6-v2 (local, no API key, 384-dim) per user recommendation; torch weight accepted
- ASSUMED — embed content only (not heading_path) at ingest; questions embedded as-is
- ASSUMED — CANDIDATES=50 fusion input depth, RRF k=60 (ranx default); k=5 eval unchanged
- ASSUMED — re-running `setup` (full re-ingest) is acceptable
- ASSUMED — per-question fuse calls (150 total) OK; numba JIT compiles once (~seconds), cached
- Question: CPU-only torch vs full CUDA wheel? Default full; slim optional.
- Question: fuse ts_rank_cd INTO the rrf mix too (3-way)? Request says semantic + bm25 only → 2-way.

## Notes

- Rejected: `ranx.fusion.rrf(runs, k=60)` direct call — `ranx.fuse(..., method="rrf", params={"k": 60})` is the documented top-level API; same dispatch, keeps `norm` explicit None.
- Rejected: batch fusion (one Run per whole golden set per table) — fewer fuse calls but breaks the per-question closure interface evaluate() already uses; 150 per-question calls cost ~nothing after JIT.
- Rejected: passing raw `list[float]` to pgvector param — psycopg dumper registered for `np.ndarray`/`Vector`, not bare lists; `model.encode()` already returns ndarray.
- Rejected: mock-embedding fallback (structural-only eval) — user flagged semantic/RRF would be noise; real embeddings required for meaningful eval.
- Risk: HNSW approximate scan at depth 50 with default `ef_search=40` → misses; mitigated with `SET hnsw.ef_search = 100` in `_semantic_ranked`.
- Risk: MAP ≡ MRR still holds (1 relevant chunk/question) — unchanged from before, fine.
- Sequencing: deps → retrievers.py → setup.py → eval.py → main.py; setup before eval needs DB running.

## Amendments

### A1 — add `cross_encoder_retriever` (cross-encoder/ms-marco-MiniLM-L-6-v2)

Change: 5th retriever — cross-encoder reranker over BM25 candidates; evals.md grows 12 → 15 rows.
Why: user request — cross-encoder relevance search over all 3 chunkers, same golden set/metrics.

**Verified facts (grounded, sentence-transformers 6.2.0 source `sentence_transformers/cross_encoder/model.py` + HF card):**

- `from sentence_transformers import CrossEncoder` — top-level export; card snippet: `model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`.
- `predict(inputs: Sequence[tuple[str, str]], batch_size=32, show_progress_bar=None) -> np.ndarray` — one relevance score per (query, passage) pair, higher = more relevant (card: "sort the passages in a decreasing order"; docstring example scores 10.67 > -6.47).
- `num_labels=1` (default for this model) → default `activation_fn=nn.Sigmoid` — monotonic, order-preserving → ranking identical to raw logits. Not a concern.
- Alternative API `rank(query, documents, top_k)` returns `[{"corpus_id": idx, "score": s}]` — sorted, but `corpus_id` is positional into `documents`; `predict` + own sort (with id tie-break) chosen instead, matches project's deterministic sort pattern (eval.py:57).
- Model: MiniLM-L-6 cross-encoder, ~22M params (same family as all-MiniLM-L-6-v2), max seq 512 tokens — chunks ≤800 chars ≈ 200 tokens, safe.
- Cross-encoders score every (query, passage) pair — NO ANN index, cannot "search" alone. Needs a candidate pool. **ASSUMED — candidates = BM25 top-CANDIDATES(50)** (standard two-stage retrieve→rerank; reuses existing `_bm25_ranked` cache; 50 pairs × 50 questions × 3 tables = 7.5k pairs, CPU seconds). Rejected: full-corpus scoring (600 chunks × 150 queries × 3 = 270k pairs, CPU minutes-to-hours); union(bm25, semantic) candidates (heavier, couples CE to semantic model).

**retrievers.py additions (extends A0 design):**

```python
CROSS_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_CROSS = None
def _get_cross_encoder():
    # lazy singleton, same pattern as _get_model
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder
        _CROSS = CrossEncoder(CROSS_MODEL_NAME)
    return _CROSS

# AMEND _bm25_ranked cache entry: cache[table] = (ids, contents, bm25)   # WAS (ids, bm25)
def _bm25_ranked(conn, table: str, cache: dict) -> Callable[[str, int], list[tuple[int, float]]]
    # unchanged return shape (id, bm25_score) — closure now unpacks ids, _, bm25 = cache[table]

def _bm25_corpus(conn, table: str, cache: dict) -> tuple[list[int], list[str]]
    # NEW: triggers same cache entry; returns parallel (ids, contents)
    # if table not in cache: SELECT id, content FROM {table} → ids, contents
    #   then build BM25Okapi (same tokenize) → cache[table] = (ids, contents, bm25)
    # shared by _bm25_ranked, cross_encoder_retriever — corpus fetched ONCE per table

def cross_encoder_retriever(conn, table: str, cache: dict) -> Callable[[str], list[int]]
    # ids, contents = _bm25_corpus(conn, table, cache)        # id -> content via dict(zip(ids, contents))
    # pos = {id: i for i, id in enumerate(ids)}
    # ce = _get_cross_encoder(); ranked = _bm25_ranked(conn, table, cache)
    # content_of = dict(zip(ids, contents))
    # closure(question, k=K):
    #   cands = ranked(question, CANDIDATES)                  # [(id, bm25_score)] best-first
    #   if not cands: return []
    #   pairs = [(question, content_of[id]) for id, _ in cands]
    #   scores = ce.predict(pairs, batch_size=32)             # np.ndarray, higher = more relevant
    #   order = sorted(range(len(cands)), key=lambda i: (-float(scores[i]), cands[i][0]))[:k]
    #   return [cands[i][0] for i in order]
```

**eval.py amendment (run() retrievers dict):**

```python
retrievers = {   # row order: ts_rank_cd, bm25, semantic, rrf, cross_encoder × (fixed, recursive, markdown)
    "ts_rank_cd": ts_retriever(conn, table),
    "bm25": bm25_retriever(conn, table, cache),
    "semantic": semantic_search_retriever(conn, table),
    "rrf": rrf_retriever(conn, table, cache),
    "cross_encoder": cross_encoder_retriever(conn, table, cache),
}
```

**Task changes:**

- Task 2 (CREATE `retrievers.py`): add `_get_cross_encoder`, `_bm25_corpus`, `cross_encoder_retriever`; widen `_bm25_ranked` cache entry to `(ids, contents, bm25)`.
  - GOTCHA: CE first call downloads model (~90MB) — same one-time cost as bi-encoder, cached ~/.cache/huggingface.
  - GOTCHA: `predict` returns ndarray; float() cast per element before sort; tie-break by chunk id (deterministic, matches bm25 pattern eval.py:57).
  - VALIDATE: `uv run python -c "from retrievers import cross_encoder_retriever; print('ok')"`
- Task 4 (UPDATE `eval.py`): import + 5-entry dict. VALIDATE: evals.md **15 rows**.
- Task 6 expectations: 15 rows; cross_encoder rows should beat bm25 rows on most chunkers (reranking usually does) — soft expectation, not gate.
- No new deps — sentence-transformers already added in Task 1; CrossEncoder same package.

**Acceptance adds:**

- [ ] evals.md 15 rows, stable order (ts_rank_cd, bm25, semantic, rrf, cross_encoder) × (fixed, recursive, markdown)
- [ ] cross-encoder candidate pool = BM25 top-50, reranked, top-5 returned; corpus fetched once per table (cache)

**Open:**

- ASSUMED — CE candidates from BM25 top-50 (not union with semantic, not full corpus). Confirm or switch.
- ASSUMED — cross_encoder row placed last in evals.md (after rrf). Alternative: group rerankers together.
