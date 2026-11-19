# Plan: RRF→cross-encoder rerank row + `pipeline` command (contextual retrieval)

Type: enhance | Complexity: H | Systems: Postgres (pgvector/HNSW/GIN), ollama (local LLM), sentence-transformers CrossEncoder, Python 3.12/uv
Builds on: `.pi/plans/semantic-rrf-retrievers.md` (implemented; its Amendment A1 is SUPERSEDED by this plan — A1 used ms-marco-MiniLM-L-**6**-v2 reranking BM25 top-50; this plan uses ms-marco-MiniLM-L-**12**-v2 reranking RRF top-10)
Deps to add: `ollama` (0.6.3; deps httpx>=0.27, pydantic>=2.9 — light, no torch)

## Goal

As dev, I want (a) a 5th eval retriever — cross-encoder reranking of RRF's top-10 candidates → top-5 — evaluated across the existing 3 chunker tables (evals.md 12 → 15 rows), and (b) a new `pipeline` Typer command that evaluates ONE end-to-end pipeline (markdown chunking → Anthropic-style contextual retrieval via local `qwen3.5:4b` → BM25 → semantic → RRF → cross-encoder rerank) on the same golden set, so I can measure how much contextual retrieval + rerank adds over plain markdown.

## Out of scope

- Not: changing K (=5), CANDIDATES (=50), RRF_K (=60), golden set, relevance/skip logic, scorers.py (untouched)
- Not: changing existing 4 retriever rows — the 12 current evals.md rows must reproduce EXACTLY
- Not: `setup` command growing a 4th table — `setup` semantics unchanged (pipeline builds its own table, see D7)
- Not: reranking the `rrf` row itself (rrf stays top-5); top-10 out-of-fusion depth applies ONLY to the rrf_ce candidate pool
- Not: contextual retrieval for the rrf_ce eval row — requirement 2 uses the EXISTING tables only; only `pipeline` uses contexts. Explicit split: evals.md rrf_ce rows = plain tables; pipeline.md = contextual table
- Not: pytest (project convention: inline `uv run` validation)
- Not: ranx anywhere except RRF fusion; not `rank()` / softmax APIs of CrossEncoder

## Verified facts (grounded)

### Repo

- `retrievers.py` current: `K=5`, `RRF_K=60`, `CANDIDATES=50` (retrievers.py:12-14); `_get_model()` lazy singleton (retrievers.py:22); `_bm25_ranked(conn, table, cache)` builds cache entry `(ids, BM25Okapi)` from `SELECT id, content` (retrievers.py:60-69); `_semantic_ranked(conn, table, model)` reads ONLY `id` + `embedding` — never reads content (retrievers.py:82-100) → contextual variant needs NO change to `_semantic_ranked`, only different embedded vectors at ingest; `rrf_retriever` guards empty runs, `fuse(..., norm=None, method="rrf", params={"k": 60})`, sorts `(-fused, int(doc_id))` (retrievers.py:106-125).
- `eval.py`: `relevant_ids(rows, snippet)` = normalized snippet containment in content + 60-char prefix fallback (eval.py:24-33); `evaluate(retriever, golden, relevant, k=K)` retriever-agnostic (eval.py:36-59); `run()` loops `TABLES` (setup.py:24) × 4-retriever dict, writes evals.md (eval.py:62-101). Pipeline can import `load_golden`, `relevant_ids`, `evaluate`, `K` — NO eval.py refactor needed.
- `setup.py`: DDL has generated tsv = `setweight(heading_path,'A') || setweight(content,'B')` (setup.py:29-35); `bulk_insert` embeds content at batch_size=64 with shape assert (setup.py:62-79); `connect()` registers pgvector dumpers (setup.py:39-42). `BOOKS` = 2 books: `~/Desktop/letsgo_further.md`, `~/Desktop/cloud_run.md` (setup.py:16-20).
- DB live: `markdown_chunks` = 1740 rows (Google Cloud Run 568, Let's Go Further 1172). Golden set = 50 questions, all with `ground_truth_snippet`. Current skipped: fixed 3, recursive 0, markdown 0.
- Docs on disk: `letsgo_further.md` = 765,022 chars / 114,954 words (~150k tokens @~1.3 tok/word); `cloud_run.md` = 471,700 chars / 57,053 words (~74k tokens). Whole-document prompts are LONG — drives D2 design.
- CE candidate-content need: RRF fusion yields ids only; CE scores (question, passage) pairs → must widen BM25 cache to carry bodies (already sketched in superseded A1 as `(ids, contents, bm25)`).

### Cross-encoder model (`cross-encoder/ms-marco-MiniLM-L12-v2`) — HF card + config.json (fetched 2025)

- BERT, 12 layers, hidden 384, `max_position_embeddings: 512`, `num_labels` 1 (`id2label: {"0": "LABEL_0"}`), `sbert_ce_default_activation_function: torch.nn.modules.linear.Identity` → `predict` returns RAW LOGITS (can be negative; card example `9.22` vs `-4.08`). Monotonic — ranking only, unaffected. (Differs from A1's L-6 note; verified from this model's config.)
- `model.safetensors` = 133,469,020 B ≈ **127MB** one-time download (HF API file listing). Card usage: `CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')`, `model.predict([(q, p1), (q, p2)])` — "sort the passages in a decreasing order". Card MRR@10 (MS Marco dev) 39.02, 960 docs/s on V100 — for reference; ~2× slower than L-6-v2.
- 512-token budget check: chunks ≤ 800 chars ≈ ~200 tokens + question ~20 tokens → fits with room; no truncation concern (but ST truncates at 512 anyway).
- sentence-transformers **6.2.0** (installed, verified via `uv run python -c`): `CrossEncoder.predict(inputs, prompt_name=None, prompt=None, batch_size=32, show_progress_bar=None, activation_fn=None, apply_softmax=False, convert_to_numpy=True, ...) -> np.ndarray` (convert_to_numpy=True default); `CrossEncoder.__init__(..., max_length=None, ...)` — default max_length from model config (512). Top-level export `from sentence_transformers import CrossEncoder` confirmed.

### Ollama Python client (v0.6.3, source fetched from ollama-python repo v0.6.3 tag)

- `ollama.chat(model=..., messages=[{'role': 'user', 'content': prompt}], options={...}, think=...)` — module-level function wraps default `Client(host='http://localhost:11434')` (README + `_client.py:344-355`). `think: Optional[bool|str]` param exists in 0.6.3.
- Response: `response.message.content` (or `response['message']['content']`) — `ChatResponse` pydantic type (README usage).
- `options` dict → `Options` (ollama `_types.py:104-125`): `temperature: Optional[float]`, `num_predict: Optional[int]`, `num_ctx: Optional[int]`, `seed`, `top_k`, `top_p` — pass `{"temperature": 0, "num_predict": 128, "num_ctx": CONTEXT_NUM_CTX}`.
- `ollama.list()` → `ListResponse` with `.models` (each has `.model` name) — use for reachability + model-present checks. Server down → httpx `ConnectError`; API error → `ollama.ResponseError` (`_client.py:147`).
- `qwen3.5:4b` (ollama library page, fetched): arch `qwen35`, 4.21B params, Q4_K_M (2.6GB) + vision projector; tags show **thinking** capability → pass `think=False` to keep output to the succinct context only.
- `qwen3.5:4b` context window: `Qwen/Qwen3.5-4B` HF `config.json` `text_config.max_position_embeddings = 262144` (256k). Hybrid attention: 32 layers, only every 4th is full attention (8 full-attn layers), 4 KV heads × head_dim 256 → K+V ≈ 32KB/token → **KV ≈ 8.4GB per slot at num_ctx=262144** (f16). Ollama allocates KV per parallel slot; `OLLAMA_NUM_PARALLEL` defaults to auto (up to 4) → worst case ~34GB. Mitigations in Notes.

## Decisions (design problems, resolved)

**D1 — Storage**: new table `markdown_context_chunks`, built/owned by `pipeline.py`, NOT by `setup.py`. Same columns as `markdown_chunks` PLUS `context TEXT` (nullable). `content` keeps the ORIGINAL chunk text verbatim (copied from `markdown_chunks`) → `relevant_ids` containment unaffected; context lives in `context` column. Concatenation is canonical in ONE helper + one SQL expr:
- Python: `_combined(context, content) -> f"{context}\n{content}"` (pipeline.py) — embedding input, CE passage for pipeline.
- SQL body expr (BM25 + tsv): `coalesce(context, '') || E'\n' || content`.
- tsv for the contextual table: `setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') || setweight(to_tsvector('english', coalesce(context, '') || E'\n' || content), 'B')` — to_tsvector with explicit `'english'` regconfig is immutable → legal in GENERATED ALWAYS AS.
- Helpers select bodies via a `body_expr: str = "content"` parameter (SQL expression, interpolated from internal constants — never user input; existing code already f-string-interpolates `{table}` the same way). `_semantic_ranked` untouched (embedding column already encodes whatever was embedded at backfill).

**D2 — Ollama calls**: concurrent via `ThreadPoolExecutor(max_workers=CONTEXT_WORKERS)`, contexts cached in the `context` column (skip rows `WHERE context IS NOT NULL`) — chosen over a separate contexts table because one table keeps embed+tsv+search consistent and the pipeline self-contained. `options={"temperature": 0, "num_predict": 128, "num_ctx": CONTEXT_NUM_CTX}` with `CONTEXT_NUM_CTX = 262144` (covers ~150k-token letsgo doc + chunk + 128 out), `think=False`. Commit UPDATEs in batches of 32 so a crash keeps completed work. Tradeoffs (KV memory, prefix caching) in Notes.

**D3 — Decomposition** (eval rows + pipeline share code):
- `_rrf_ranked(conn, table, cache, body_expr="content")` → closure `(question, depth) -> [(id, fused_score)]` — extracted from today's `rrf_retriever` body verbatim. `rrf_retriever` becomes a thin wrapper returning top-K ids (existing 12 rows reproduce).
- `_get_cross_encoder()` + `CROSS_MODEL_NAME` — lazy singleton mirroring `_get_model()` (retrievers.py:22).
- `_cross_encoder_rerank(ce, question, candidates, k)` — pure: pairs → predict → sort `(-score, id)` → top-k ids. Both eval row and pipeline call it.
- `rrf_ce_retriever(conn, table, cache, body_expr="content")` → closure `(question, k=K) -> top-5 ids`: RRF top-`RRF_CE_CANDIDATES`=10 → id→body lookup → CE rerank → 5.
- New constant `RRF_CE_CANDIDATES = 10` (out-of-fusion depth). `CANDIDATES=50` stays (per-leg depth INTO fusion).

**D4 — cache widening**: `cache[table] = (ids, bodies, bm25)` where `bodies` = result of `body_expr` (raw `content` for plain tables; context+content for the contextual table — cache holds the SELECTED body, which is exactly what BM25 corpus AND CE passage need). New `_bm25_corpus(conn, table, cache, body_expr)` returns `(ids, bodies)` and triggers the cache build; `_bm25_ranked` reuses it. Keyed by table only — one table is always queried with one body_expr in this codebase (GOTCHA below).

**D5 — model**: ms-marco-MiniLM-L12-v2 per requirement (NOT A1's L-6). `predict(pairs, batch_size=32)` → ndarray raw logits, higher = more relevant; verified above.

**D6 — pipeline eval scope**: relevance = snippet containment in ORIGINAL `content` column of `markdown_context_chunks` (context column excluded — another reason context is a separate column). Reuse `eval.load_golden`, `eval.relevant_ids`, `eval.evaluate` via import; no eval.py refactor. Skip count expected: 0 (content identical to `markdown_chunks`, which has 0 skipped).

**D7 — main.py**: `pipeline` builds/ensures its own table idempotently (CREATE TABLE IF NOT EXISTS + copy rows missing by (book_title, chunk_index) + generate missing contexts + backfill NULL embeddings + CREATE INDEX IF NOT EXISTS HNSW) — `setup` unchanged. Output: print table AND write `pipeline.md` (separate file; evals.md stays the 15-row eval artifact). Row shape mirrors evals.md: `| rrf_ce | markdown_context | recall@5 | MRR | MAP |`.

**D8 — naming**: eval row key `rrf_ce` (matches existing lowercase keys: ts_rank_cd, bm25, semantic, rrf; user's "RRF + Cross Encoder Rerank" is the human gloss, documented in docstrings). Pipeline table chunker label: `markdown_context`.

## File layout

- `retrievers.py` — UPDATE: constants + cross-encoder singleton + cache widening + `_rrf_ranked` extraction + `rrf_ce_retriever`
- `eval.py` — UPDATE: import + 5th dict entry → 15 rows
- `pipeline.py` — NEW: contextual table DDL/populate, ollama context generation, embedding backfill, HNSW, single-pipeline eval, pipeline.md output
- `main.py` — UPDATE: `pipeline` command
- `pyproject.toml` / `uv.lock` — UPDATE: `uv add ollama`
- `evals.md` — OUTPUT: 15 rows (existing 12 unchanged + 3 rrf_ce rows)
- `pipeline.md` — OUTPUT: 1 row (the pipeline)

## retrievers.py — exact changes

```python
# constants (top, after MODEL_NAME)
RRF_CE_CANDIDATES = 10  # RRF out-of-fusion depth feeding the cross-encoder
CROSS_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"  # BERT, 512 max tokens, ~127MB

_CROSS = None
def _get_cross_encoder():
    """Lazy singleton, same pattern as _get_model (retrievers.py:22)."""
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder
        _CROSS = CrossEncoder(CROSS_MODEL_NAME)
    return _CROSS

# WIDENED: _bm25_ranked(conn, table, cache, body_expr: str = "content")
#   -> closure(question, depth) -> [(id, bm25_score)]   (return shape UNCHANGED)
# internals:
#   if table not in cache:
#       cur.execute(f"SELECT id, {body_expr} FROM {table}")
#       ids, bodies = [r[0] for r in rows], [r[1] for r in rows]
#       cache[table] = (ids, bodies, BM25Okapi([_tokenize(b) for b in bodies]))
#   ids, _, bm25 = cache[table]
#   ranked() unchanged (scores over _tokenize(question), sort (-score, i)[:depth])

def _bm25_corpus(conn, table, cache, body_expr: str = "content") -> tuple[list[int], list[str]]:
    """(ids, bodies) for the table's selected body; triggers the shared cache entry."""

def _rrf_ranked(conn, table, cache, body_expr: str = "content") -> Callable[[str, int], list[tuple[int, float]]]:
    """Extracted VERBATIM from rrf_retriever body (retrievers.py:112-124):
       sem = _semantic_ranked(conn, table, _get_model())   # body_expr irrelevant: reads embedding
       bm  = _bm25_ranked(conn, table, cache, body_expr)
       closure(question, depth):
           ... empty-run guards, Run.from_dict x2, fuse(norm=None, method="rrf", params={"k": RRF_K}),
           sorted (-fused, int(doc_id)) -> [(int(doc_id), fused_score), ...][:depth]
    """

def rrf_retriever(conn, table, cache) -> Callable[[str], list[int]]:  # UNCHANGED behavior
    ranked = _rrf_ranked(conn, table, cache)
    def retrieve(question, k=K): return [id for id, _ in ranked(question, k)]
    return retrieve

def _cross_encoder_rerank(ce, question, candidates: list[tuple[int, str]], k: int) -> list[int]:
    """candidates = [(id, body)] in RRF order; predict raw logits; top-k."""
    # pairs = [(question, body) for _, body in candidates]
    # scores = ce.predict(pairs, batch_size=32)            # np.ndarray; Identity activation -> raw logits
    # order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i][0]))[:k]
    # return [candidates[i][0] for i in order]

def rrf_ce_retriever(conn, table, cache, body_expr: str = "content") -> Callable[[str], list[int]]:
    """Question -> top-5 ids: RRF(top-10 of bm25+semantic) -> cross-encoder rerank."""
    # ranked = _rrf_ranked(conn, table, cache, body_expr)
    # ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    # body_of = dict(zip(ids, bodies)); ce = _get_cross_encoder()
    # def retrieve(question, k=K):
    #     fused = ranked(question, RRF_CE_CANDIDATES)        # top-10 candidate pool
    #     if not fused: return []
    #     cands = [(id, body_of[id]) for id, _ in fused]
    #     return _cross_encoder_rerank(ce, question, cands, k)
    # return retrieve
```

## eval.py — exact changes

```python
# ADD import: from retrievers import rrf_ce_retriever
# retrievers dict gains 5th entry (insertion order = row order):
retrievers = {
    "ts_rank_cd": ts_retriever(conn, table),
    "bm25": bm25_retriever(conn, table, cache),
    "semantic": semantic_search_retriever(conn, table),
    "rrf": rrf_retriever(conn, table, cache),
    "rrf_ce": rrf_ce_retriever(conn, table, cache),   # body_expr default "content" — existing tables, NO contextual retrieval
}
# module docstring: 4 -> 5 retrievers, 12 -> 15 rows
```

Everything else in `run()` untouched (rows, table loop, skipped line, evals.md write).

## pipeline.py — NEW

```python
"""Contextual-retrieval pipeline: markdown chunks + qwen3.5:4b contexts
+ BM25 + semantic + RRF fusion + cross-encoder rerank, evaluated on the golden set.

Contextual retrieval (Anthropic recipe): before indexing, each chunk gets a
short LLM-generated context situating it in the whole document; context is
prepended for BM25 corpus, embeddings, and rerank passages. Original chunk
text stays in `content` so snippet-containment relevance is unaffected.
"""
OLLAMA_HOST = "http://localhost:11434"        # ollama-python default
CONTEXT_MODEL = "qwen3.5:4b"
CONTEXT_NUM_CTX = 262144                      # qwen3.5:4b max_position_embeddings (verified)
CONTEXT_WORKERS = 4                           # ThreadPoolExecutor max_workers
CONTEXT_MAX_PREDICT = 128                     # cap output; contexts are short
CONTEXT_COMMIT_BATCH = 32                     # UPDATE commit granularity (crash-safe cache)
PIPELINE_TABLE = "markdown_context_chunks"
BODY_EXPR = "coalesce(context, '') || E'\\n' || content"   # D1 SQL concatenation
PIPELINE_MD = Path("pipeline.md")

PROMPT_TEMPLATE = """<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

_DDL = """
CREATE TABLE IF NOT EXISTS markdown_context_chunks (
    id SERIAL PRIMARY KEY,
    book_title TEXT NOT NULL,
    chunk_index INT NOT NULL,
    heading_path TEXT,
    content TEXT NOT NULL,          -- ORIGINAL chunk text (relevance containment)
    context TEXT,                   -- LLM context; NULL until generated
    embedding VECTOR(384),          -- NULL until backfilled over context+content
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(context, '') || E'\n' || content), 'B')
    ) STORED
);
CREATE INDEX IF NOT EXISTS markdown_context_chunks_tsv_gin
    ON markdown_context_chunks USING GIN (tsv);
"""
# NOTE: no DROP — table holds the expensive LLM cache; regenerating means manually
# dropping the table (documented in --help/docstring).

def _connectivity() -> None:
    # ollama.list() inside try/except httpx.ConnectError, ollama.ResponseError:
    #   typer.echo("ollama not reachable at {OLLAMA_HOST} — start it (ollama serve) and pull the model (ollama pull qwen3.5:4b)", err=True); raise typer.Exit(1)
    # also: model not in [m.model for m in resp.models] -> same exit path with pull instruction

def _doc_texts() -> dict[str, str]:
    # {book_title: full markdown text} from setup.BOOKS paths; missing file -> typer.Exit(1)

def ensure_table(conn) -> None:
    # execute _DDL; then INSERT missing rows:
    #   INSERT INTO {PIPELINE_TABLE} (book_title, chunk_index, heading_path, content)
    #   SELECT m.book_title, m.chunk_index, m.heading_path, m.content FROM markdown_chunks m
    #   WHERE NOT EXISTS (SELECT 1 FROM {PIPELINE_TABLE} c
    #                     WHERE c.book_title = m.book_title AND c.chunk_index = m.chunk_index);

def _one_context(client, doc_text: str, chunk: str, row_id: int) -> tuple[int, str]:
    # prompt = PROMPT_TEMPLATE.replace("{{WHOLE_DOCUMENT}}", doc_text).replace("{{CHUNK_CONTENT}}", chunk)
    # resp = client.chat(model=CONTEXT_MODEL, messages=[{"role": "user", "content": prompt}],
    #                    options={"temperature": 0, "num_predict": CONTEXT_MAX_PREDICT,
    #                             "num_ctx": CONTEXT_NUM_CTX},
    #                    think=False)
    # return row_id, resp.message.content.strip()
    # per-call retry (1x) on ollama.ResponseError/httpx errors; persistent failure -> raise

def generate_contexts(conn) -> None:
    # _connectivity(); docs = _doc_texts()
    # SELECT id, book_title, content FROM {PIPELINE_TABLE} WHERE context IS NULL  (ordered by book_title, chunk_index)
    # if rows: ThreadPoolExecutor(max_workers=CONTEXT_WORKERS) -> _one_context per row
    #   futures consumed as_completed; buffer UPDATE ... SET context = %s WHERE id = %s
    #   executemany + commit every CONTEXT_COMMIT_BATCH rows (and at end)
    #   progress print every 100 rows (n/total, rate)
    # chunk body passed to LLM = ORIGINAL content (the chunk being situated) — NOT context+content

def backfill_embeddings(conn) -> None:
    # model = _get_model()   # from retrievers; same 384-dim bi-encoder as setup.bulk_insert (setup.py:62-79)
    # loop: SELECT id, context, content FROM {PIPELINE_TABLE} WHERE embedding IS NULL
    #   texts = [f"{context}\n{content}" for ...]      # D1 Python concatenation
    #   model.encode(texts, batch_size=64)             # chunk the SELECT into e.g. 512-row batches
    #   executemany UPDATE ... SET embedding = %s WHERE id = %s ; commit
    # CREATE INDEX IF NOT EXISTS markdown_context_chunks_hnsw
    #   ON markdown_context_chunks USING hnsw (embedding vector_cosine_ops)
    #   WITH (m = 16, ef_construction = 64);  then ANALYZE;

def run() -> None:
    # golden = load_golden(); cache = {}
    # with connect() as conn:
    #     ensure_table(conn); generate_contexts(conn); backfill_embeddings(conn)
    #     SELECT id, content FROM {PIPELINE_TABLE}   (content ONLY — original text)
    #     relevant = [relevant_ids(rows, q["ground_truth_snippet"]) for q in golden]
    #     retriever = rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)
    #     m = evaluate(retriever, golden, relevant)      # eval.evaluate, k=K=5
    # skipped = count of empty relevant sets (expect 0)
    # table (same column shape as evals.md):
    #   | Retriever | Chunker | recall@5 | MRR | MAP |
    #   | rrf_ce | markdown_context | ... |
    # write pipeline.md + print (mirrors eval.py:86-99)
```

Imports: `from eval import K, evaluate, load_golden, relevant_ids`, `from retrievers import _get_model, rrf_ce_retriever`, `from setup import BOOKS, connect`.

## main.py

```python
# ADD: from pipeline import run as run_pipeline
@app.command()
def pipeline() -> None:
    """Evaluate the contextual-retrieval pipeline (markdown + qwen3.5:4b contexts
    + BM25 + semantic + RRF + cross-encoder rerank); builds markdown_context_chunks
    idempotently (contexts cached — second run is fast); writes pipeline.md."""
    run_pipeline()
# eval_command docstring: 4 -> 5 retrievers, 12 -> 15 rows
```

## Tasks (dependency order)

### 1. UPDATE `pyproject.toml`

- DO: `uv add ollama`
- GOTCHA: ollama pulls httpx + pydantic only — light; no torch interaction.
- VALIDATE: `uv run python -c "import ollama; print(ollama.__version__)"`
- MEETS: ollama client available.

### 2. UPDATE `retrievers.py` (depends on 1? no — CE needs only sentence-transformers already installed; independent of 1)

- DO: add `RRF_CE_CANDIDATES`, `CROSS_MODEL_NAME`, `_get_cross_encoder`, `_bm25_corpus`; widen `_bm25_ranked` to `body_expr` param + `(ids, bodies, bm25)` cache; extract `_rrf_ranked`; rewrite `rrf_retriever` as wrapper; add `_cross_encoder_rerank`, `rrf_ce_retriever`.
- PATTERN: `_get_model` lazy singleton — retrievers.py:19-27; cache build — retrievers.py:60-69; deterministic sort `(-score, i)` — retrievers.py:71-73.
- GOTCHA: `rrf_retriever` behavior must be bit-identical (existing 12 rows reproduce) — extraction is move-verbatim, no logic edits; fused closure returns (id, score) so wrapper `[:k]` gives old top-5.
- GOTCHA: cache keyed by `table` only; each table used with exactly one body_expr in this codebase (plain tables → "content"; contextual → BODY_EXPR). Never query one table with two body_exprs in one cache.
- GOTCHA: `predict` returns raw logits (Identity activation, can be negative) — never compare to 0 or softmax; `float()` cast before sort.
- GOTCHA: first `_get_cross_encoder()` downloads ~127MB once (HF cache).
- GOTCHA: `body_expr` is SQL text interpolated into f-string — internal constants only, never user input (same trust level as existing `{table}` interpolation, retrievers.py:64,88).
- VALIDATE: `uv run python -c "from retrievers import rrf_ce_retriever, _get_cross_encoder, _rrf_ranked; print('ok')"` then (DB running) `uv run python -c "
from setup import connect
from retrievers import rrf_ce_retriever
with connect() as conn:
    r = rrf_ce_retriever(conn, 'markdown_chunks', {})
    ids = r('How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?')
    print(ids); assert len(ids) == 5 and all(isinstance(i, int) for i in ids)"`
- MEETS: requirement 1 (top-10 pool) + requirement 2 machinery.

### 3. UPDATE `eval.py` (depends on 2; tables already in DB)

- DO: import `rrf_ce_retriever`; add `"rrf_ce": rrf_ce_retriever(conn, table, cache),` as 5th dict entry; docstring 15 rows.
- PATTERN: retrievers dict — eval.py:83-88.
- GOTCHA: rrf_ce row uses EXISTING tables (body_expr default "content") — no contextual retrieval here.
- GOTCHA: save current results first: `cp evals.md /tmp/evals.before.md` — diff after run; existing 12 rows must be byte-identical.
- VALIDATE: `cp evals.md /tmp/evals.before.md && uv run python main.py eval && diff <(head -16 /tmp/evals.before.md) <(head -16 evals.md); grep -c '^| rrf_ce' evals.md` → diff empty (first 12 data rows byte-identical), grep prints 3 (fixed/recursive/markdown rrf_ce rows).
- MEETS: evals.md 15 rows; existing 12 reproduce.

### 4. CREATE `pipeline.py` (depends on 1, 2)

- DO: per design above — DDL, `_connectivity`, `_doc_texts`, `ensure_table`, `_one_context`, `generate_contexts`, `backfill_embeddings`, `run`.
- PATTERN: DDL/generated tsv — setup.py:29-35; HNSW build — setup.py:83-96; bulk encode + executemany — setup.py:62-79; eval-table write — eval.py:86-99.
- GOTCHA: prompt template substitutions BEFORE the `<document>` tag too — template is exact per user request; whole document = full book markdown from `setup.BOOKS` path, keyed by the row's `book_title`.
- GOTCHA: options `num_ctx` REQUIRED — ollama default context (~4k) would silently truncate the whole-document prompt; `CONTEXT_NUM_CTX = 262144` matches the model's 262144 max_position_embeddings (verified from HF config). All requests use the SAME num_ctx (mixed values force ollama reloads).
- GOTCHA: KV memory: ~8.4GB per parallel slot at num_ctx=262144 (verified math in Facts). If ollama OOMs: reduce `CONTEXT_WORKERS` to 1-2, and/or start ollama with `OLLAMA_NUM_PARALLEL=1`, optionally `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` (~halves KV). Sequential-with-prefix-cache note in Notes.
- GOTCHA: `think=False` — qwen3.5 is a thinking-capable model; without this, `num_predict=128` can be consumed by reasoning and `.message.content` empty while thinking lands in `.message.thinking`.
- GOTCHA: rows must be grouped by book when submitting (order by book_title, chunk_index) so ollama's per-slot prompt-prefix cache reuses the shared document prefix across that book's chunks.
- GOTCHA: commit in batches of 32 — 1740 LLM calls is long; a crash must not lose all completed contexts. Re-running `pipeline` resumes (WHERE context IS NULL).
- GOTCHA: embeddings backfilled AFTER all contexts done (embedding input = `f"{context}\n{content}"`); `WHERE embedding IS NULL` keeps re-runs cheap if interrupted mid-backfill.
- GOTCHA: tsv generated column auto-updates when `context` is UPDATEd — no manual reindex; GIN index maintained incrementally.
- GOTCHA: DO NOT drop the table on re-run — cache is the point. `CREATE TABLE IF NOT EXISTS` + `INSERT ... WHERE NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` only.
- GOTCHA: `_doc_texts()` file-missing → clean exit (typer.Exit(1)) with the missing path in the message, not a stack trace.
- VALIDATE (ollama + DB running, model pulled): `uv run python -c "
from setup import connect
from pipeline import ensure_table, generate_contexts
with connect() as conn:
    ensure_table(conn); generate_contexts(conn)
" && psql postgres://postgres:password@localhost/p2 -c "SELECT count(*) FILTER (WHERE context IS NULL) AS no_ctx, count(*) AS total, count(*) FILTER (WHERE embedding IS NULL) AS no_emb FROM markdown_context_chunks;"` → no_ctx 0, total 1740. (First run is SLOW — see Notes for expected runtime.)
- VALIDATE (embeddings + eval): `uv run python main.py pipeline` → prints 1-row table, writes pipeline.md; `psql ... -c "SELECT count(*) FROM markdown_context_chunks WHERE embedding IS NULL;"` → 0.
- VALIDATE (cache): `time uv run python main.py pipeline` twice — second run skips generation (fast; only eval cost).
- MEETS: requirements 3, 4; D2 caching.

### 5. UPDATE `main.py` (depends on 4)

- DO: `pipeline` command; `eval` docstring 15 rows.
- VALIDATE: `uv run python main.py --help` shows `setup`, `eval`, `pipeline`.
- MEETS: requirement 3 CLI.

### 6. Full validation (depends on all)

- DO: end-to-end fresh check.
- VALIDATE: `uv run python main.py eval && uv run python main.py pipeline`
- Expect: evals.md 15 rows, order (ts_rank_cd, bm25, semantic, rrf, rrf_ce) × (fixed, recursive, markdown); first 12 rows match pre-change evals.md; skipped line unchanged (fixed 3, recursive 0, markdown 0); pipeline.md 1 row `| rrf_ce | markdown_context | r | m | a |`, skipped 0.
- Sanity: rrf_ce rows should generally be ≥ rrf rows on recall@5/MRR (reranking usually helps; soft expectation, not a gate). Pipeline row ideally ≥ markdown rrf_ce row (contextual retrieval gain; soft).

## Tests (project convention: inline validation, no pytest)

- Unit seams: `_cross_encoder_rerank` is pure — manual check: `uv run python -c "from retrievers import _get_cross_encoder, _cross_encoder_rerank; ce=_get_cross_encoder(); print(_cross_encoder_rerank(ce, 'q', [(1,'irrelevant text'), (2,'Berlin had a population of 3,520,031')], 2))"` → `[2, 1]`.
- Edge: empty BM25/semantic runs (already guarded in `_rrf_ranked` — extraction must preserve guards; rrf_ce handles `fused == []` → `[]`).
- Edge: table with fewer than 10 rows in fused output → `predict` on fewer pairs — works (predict takes any-length sequence).
- Integration: task 2 VALIDATE (markdown_chunks end-to-end), task 3 (15 rows), task 4 (contextual table end-to-end + resume).
- Determinism: temperature 0 + id tie-break sort → repeated runs give identical metric rows.

## Validation summary (real commands)

- L1: `uv run python -c "import retrievers, pipeline, eval, main"` (no lint config in repo)
- L2: `uv run python main.py --help`
- L3: task 2/4 VALIDATEs (retriever smoke, rerank unit seam, context coverage SQL)
- L4: `uv run python main.py eval` (15 rows; 12 reproduce) + `uv run python main.py pipeline`
- L5 manual: run pipeline twice, confirm second run fast (contexts cached); confirm `ollama serve` + `ollama pull qwen3.5:4b` precondition; kill ollama and confirm `pipeline` exits with the friendly message.

## Acceptance

- [ ] evals.md has 15 rows: (ts_rank_cd, bm25, semantic, rrf, rrf_ce) × (fixed, recursive, markdown) in that order; existing 12 rows byte-identical to pre-change
- [ ] `rrf_ce` = RRF fusion of semantic+BM25 (50 in, CANDIDATES) → top-10 → ms-marco-MiniLM-L-12-v2 rerank → top-5; uses existing tables, NO contextual retrieval
- [ ] `uv run python main.py pipeline` evaluates markdown + contextual retrieval + bm25 + semantic + rrf + cross-encoder end-to-end, prints + writes pipeline.md with recall@5, MRR, MAP
- [ ] `markdown_context_chunks`: 1740 rows, original content in `content`, context in `context`, embeddings over `context + "\n" + content`, tsv over same, GIN + HNSW indexes
- [ ] Second `pipeline` run skips context generation (cached); interrupted runs resume (context IS NULL / embedding IS NULL)
- [ ] Ollama unreachable or model not pulled → clean error + exit 1, not stack trace
- [ ] ranx still used ONLY for RRF fusion; scorers.py untouched; `setup` unchanged (3 tables)
- [ ] Pipeline relevance = snippet containment in ORIGINAL content column; skipped expected 0

## Open / assumed

- ASSUMED — ollama running at localhost:11434 with `qwen3.5:4b` pulled (user-stated); DB running with the 3 tables ingested (existing assumption)
- ASSUMED — eval row name `rrf_ce`; pipeline chunker label `markdown_context`; pipeline output file `pipeline.md` (evals.md NOT touched by `pipeline`)
- ASSUMED — whole-document prompt kept verbatim (requirement), `num_ctx=262144`; if hardware can't hold the KV, fallback = `OLLAMA_NUM_PARALLEL=1` + `CONTEXT_WORKERS=1` (still correct, slower); truncating the document would deviate from the requirement and is NOT done
- ASSUMED — first `pipeline` run is the slow path (1740 LLM calls, huge shared prefixes; rough scale in Notes) — accepted because cached forever after
- ASSUMED — contexts generated with temperature 0, num_predict 128, think=False; no retry tuning beyond 1 retry
- ASSUMED — `markdown_chunks` is stable; if source books change, drop `markdown_context_chunks` manually and re-run `pipeline` (documented in module docstring)
- Question: GPU available for ollama? CPU-only makes first-run context generation hours-scale — acceptable but worth knowing before running.

## Notes

- Superseded A1 (L-6 cross-encoder over BM25 top-50): replaced by L-12 over RRF top-10 per new request. A1's `_bm25_corpus`/cache-widening idea kept (D4); model + candidate source changed.
- Rejected: storing context prepended into `content` — breaks snippet-containment relevance and forces substring surgery everywhere; separate `context` column is one SQL expression away from combined body.
- Rejected: separate contexts table keyed (book_title, chunk_index) — two sources of truth, join at search time; single table + generated tsv + one embedding column is simpler and keeps all indexing in one place.
- Rejected: per-request `num_ctx` sized per book — ollama reloads/resizes when options differ between requests; one constant avoids thrash at the cost of headroom for the smaller book.
- Rejected: AsyncClient for ollama — ThreadPoolExecutor over sync `chat()` matches the codebase's sync style, trivially bounded; async would spread through run().
- Rejected: rerank passages = original content only for the pipeline — Anthropic's recipe reranks the context-enriched text; the whole point is the CE sees the situating context. Plain tables use plain content.
- Tradeoff — CONTEXT_WORKERS=4 vs 1: parallel slots each hold their own KV (~8.4GB at 262144 ctx for this hybrid-attention model), but after warm-up each slot reuses the shared document prefix (llama.cpp prompt cache), so per-call cost ≈ decoding ~128 output tokens. 4 workers → ~4× KV memory, ~4× throughput. If memory-bound, drop both `CONTEXT_WORKERS` and `OLLAMA_NUM_PARALLEL` together.
- Runtime scale (rough): warm-cache call ≈ 128 tokens decode; GPU ~50-100 tok/s per slot → 1740 calls / 4 workers ≈ 10-30 min one-time. Cold prefix prefill of a ~150k-token doc ≈ once per slot (~minutes each on GPU; CPU-only: hours — flagged in Open/assumed). These are estimates, not gates.
- Sequencing risk: task 3 (15-row eval) is independent of task 4 and cheap — do it first to catch the rrf_retriever refactor regressing the 12 existing rows before building the pipeline.
- `expected_source_file`/`expected_chunk_index` remain unused by the harness (pre-existing) — ids differ between markdown_chunks and markdown_context_chunks, which is fine since relevance is content-based.

## Amendments

(append-only: date — change — why)
