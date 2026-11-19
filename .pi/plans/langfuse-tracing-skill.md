# Plan: langfuse tracing skill + observed RAG ask (`traces.py`)

Implements: user task (add LangFuse skill from github.com/langfuse/skills; add `traces.py` with 4 LangFuse-observed functions) | Type: enhance | Complexity: M | Systems: Python 3.12/uv flat modules, Ollama (qwen3.5:4b), Postgres (pgvector), LangFuse v4.79.0 self-hosted (Docker, localhost:3000) | Deps to add: `langfuse>=5.2.0`, `python-dotenv`

Key adaptation (user-approved): pseudocode uses Anthropic/claude-sonnet-4-6, but says "use whatever model P2 uses". P2 uses Ollama via `ollama.Client`, model `qwen3.5:4b` = `pipeline.CONTEXT_MODEL` (pipeline.py:20). No `anthropic` dep. Generation span adapted accordingly.

## Goal

As dev, I want LangFuse observability on the RAG ask path (retrieve → rerank → generate) and the LangFuse agent skill installed at project level, so that each ask produces one inspectable trace on localhost:3000 with model, params, token usage, and per-step inputs/outputs.

## Out of scope

- Not: re-indexing books, changing retrievers/pipeline/ragas behavior, adding pytest (project has none — validation is inline `uv run` commands, per `.pi/agent-memory/implementer/MEMORY.md`).
- Not: `anthropic` dependency (project uses Ollama).
- Not: `migrate-to-langfuse` skill from upstream repo (irrelevant here).
- Not: changes to `langfuse/` (that dir is a clone of langfuse/langfuse for reference — do not touch).

## Read first (required)

- `retrievers.py` — retrieval primitives to reuse: `_rrf_ranked` (dense+BM25 RRF fusion), `_bm25_corpus`, `_get_cross_encoder`, `_cross_encoder_rerank`, constants `K`, `CANDIDATES`, `RRF_CE_CANDIDATES`.
- `pipeline.py` — `PIPELINE_TABLE`, `BODY_EXPR`, `CONTEXT_MODEL`, `OLLAMA_HOST`, `ensure_table`, `generate_contexts`, `backfill_embeddings` (traces must ensure these before retrieving).
- `ragas.py` — `ANSWER_PROMPT`, `NUM_PREDICT`, `NUM_CTX`, `answer_with_context` (context-join pattern at ragas.py:71-73), `_chat` ollama call pattern (ragas.py:47-57).
- `setup.py` — `connect()` (psycopg + register_vector), `DB_URL`.
- `main.py` — Typer command pattern to mirror for a `trace` command.
- Upstream skill source: <https://github.com/langfuse/skills> — `skills/langfuse/SKILL.md` + `skills/langfuse/references/*.md` (13 files).
- [Langfuse Python SDK v5 — instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation.md) — `@observe(as_type=...)` valid types: "generation", "span", "agent", "tool", "chain", "retriever", "embedding", "evaluator", "guardrail" (source: langfuse-python `_client/observe.py` line 121).
- [Langfuse SDK overview](https://langfuse.com/docs/observability/sdk/overview.md) — Python SDK v5 requires self-hosted server ≥ 4.59.0. Running server = Docker image `langfuse/langfuse:4` = **4.79.0** ✓ (verified locally).

## Facts established (repo)

- Project layout: flat root modules (`chunkers.py`, `setup.py`, `eval.py`, `retrievers.py`, `pipeline.py`, `ragas.py`, `variants.py`, `main.py`), Typer CLI in `main.py`, uv + pyproject.toml, Python 3.12. `traces.py` belongs at repo root.
- Model: `CONTEXT_MODEL = "qwen3.5:4b"` (pipeline.py:20); host `OLLAMA_HOST = "http://localhost:11434"` (pipeline.py:19). ollama SDK 0.6.3 installed (no `__version__` attr — use `importlib.metadata`).
- ollama 0.6.3 `ChatResponse` token fields: `prompt_eval_count`, `eval_count` (`int | None`) — verified via `ChatResponse.model_fields`.
- `hybrid_search` equivalent: `retrievers._rrf_ranked(conn, table, cache, body_expr) -> ranked(question, depth) -> list[tuple[int, float]]` (retrievers.py:135-157). Hits are **tuples `(chunk_id, fused_score)`**, NOT objects with `.id`/`.score` — pseudocode adapted to tuple unpacking.
- `cross_encoder_rerank` equivalent: `retrievers._cross_encoder_rerank(ce, question, candidates: list[tuple[int, str]], k) -> list[int]` (retrievers.py:176-186), `ce = retrievers._get_cross_encoder()`. Body text for a chunk id: `retrievers._bm25_corpus(conn, table, cache, body_expr) -> (ids, bodies)` then `dict(zip(ids, bodies))` (pattern at retrievers.py:161-163, ragas.py:71-72).
- `build_prompt` equivalent: `ragas.ANSWER_PROMPT` (context+question format string, ragas.py:28-33) formatted then sent as `[{"role": "user", "content": prompt}]` via `ollama.Client.chat(model=CONTEXT_MODEL, messages=..., options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}, think=False)` (pattern: ragas.py:47-57). Context join: `"\n\n".join(...)` (ragas.py:73). Constants `NUM_PREDICT = 512`, `NUM_CTX = 8192` (ragas.py:15-16).
- Constants: `K = 5` (retrievers.py:15), `CANDIDATES = 50` (retrievers.py:17), `RRF_CE_CANDIDATES = 10` (retrievers.py:18). Pipeline table: `markdown_context_chunks`, body expr `coalesce(context, '') || E'\n' || content` (pipeline.py:29, `BODY_EXPR` pipeline.py:27).
- `.env` already contains `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL="http://localhost:3000"` (user-added). Nothing in the repo loads `.env`; `uv run` does NOT auto-load it (verified: `LANGFUSE_BASE_URL` missing in `uv run python -c`). Langfuse Python SDK v5 reads `os.environ` only — no dotenv auto-load in SDK (verified against langfuse-python source). → traces.py must `load_dotenv()` before first `get_client()`.
- LangFuse server: Docker stack running (`langfuse-langfuse-web-1` on :3000, image `langfuse/langfuse:4` = v4.79.0 ≥ 4.59.0 required by Python SDK v5 ✓).
- `.pi/skills/` does NOT exist yet (only `.pi/{agent-memory,plans,reports}`). Global pi skills at `~/.pi/agent/skills/<name>/SKILL.md` use YAML frontmatter `name:` + `description:` (see `~/.pi/agent/skills/arch/SKILL.md`, `grill/SKILL.md`).
- `uv.lock` has no `langfuse`, no `anthropic`. git repo has zero tracked files (nothing committed yet); `.env` untracked, not gitignored.
- Upstream `skills/langfuse/SKILL.md` frontmatter: `name: langfuse`, `description: >-` (long), `allowed-tools:` list (Claude-specific — strip when installing for pi). Body links `references/*.md` (13 files: instrumentation, cli, create-dataset, prompt-engineering, prompt-migration, setting-up-evals, user-feedback, judge-calibration, error-analysis, cost-optimization, ci-cd, v4-project-migration, skill-feedback).

## Create

- `.pi/skills/langfuse/SKILL.md` — upstream skill, pi frontmatter (drop `allowed-tools`).
- `.pi/skills/langfuse/references/*.md` — copied verbatim from upstream.
- `traces.py` — 4 `@observe` functions wiring real retriever/rerank/prompt code.
- `main.py` — ADD `trace` command (ask + flush).
- `README.md` — REWRITE (currently 0 bytes) with all 6 CLI commands + explanations.
- `pyproject.toml` + `uv.lock` — ADD deps via `uv add`.
- `.gitignore` — ADD `.env` (and `langfuse/`) so keys/upstream clone are never committed.

## Patterns (real excerpts from this repo)

Factory + closure retriever (retrievers.py:159-175):

```python
def rrf_ce_retriever(conn, table: str, cache: dict, body_expr: str = "content") -> Callable[[str], list[int]]:
    ranked = _rrf_ranked(conn, table, cache, body_expr)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()
    def retrieve(question: str, k: int = K) -> list[int]:
        fused = ranked(question, RRF_CE_CANDIDATES)
        if not fused:
            return []
        cands = [(id, body_of[id]) for id, _ in fused]
        return _cross_encoder_rerank(ce, question, cands, k)
    return retrieve
```

Ollama chat call (ragas.py:47-57):

```python
resp = client.chat(
    model=CONTEXT_MODEL,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    think=False,
)
return resp.message.content.strip()
```

Typer command pattern (main.py:29-45, ragas_command).

## Tasks

Order: task 1 (skill) is Independent of tasks 2-4. Tasks 2→3→4 sequential (deps: langfuse package before import smoke; traces.py before CLI command).

### CREATE `.pi/skills/langfuse/` (skill install)

- DO: clone upstream read-only into a temp dir and copy the `langfuse` skill in:

  ```bash
  git clone --depth 1 https://github.com/langfuse/skills.git /tmp/langfuse-skills
  mkdir -p .pi/skills
  cp -r /tmp/langfuse-skills/skills/langfuse .pi/skills/langfuse
  ```

- DO: edit `.pi/skills/langfuse/SKILL.md` frontmatter — keep `name: langfuse`; keep the long `description` (collapse to a single quoted line or keep the `>-` block, both valid YAML); DELETE the `allowed-tools:` list (Claude Code-specific tool gating, meaningless in pi). Leave body and all `references/` files byte-identical (relative links keep working because the dir layout is preserved).
- GOTCHA: `~/.pi/agent/skills/` (global) vs `.pi/skills/` (project) — install at PROJECT level per task. Do NOT touch the global dir.
- GOTCHA: repo also contains a `langfuse/` dir — that is a clone of **langfuse/langfuse** (the platform, for self-host reference), NOT the skills repo. Its `.agents/skills/` are LangFuse's internal dev skills (code-review, turborepo, …) — do not install those.
- VALIDATE: `head -12 .pi/skills/langfuse/SKILL.md` shows pi-style frontmatter (`name:`, `description:`, no `allowed-tools`); `ls .pi/skills/langfuse/references/ | wc -l` → `13`.
- MEETS: skill discoverable as project skill `.pi/skills/langfuse/SKILL.md` with description frontmatter; references intact.

### ADD dependencies `langfuse`, `python-dotenv`

- DO: `uv add "langfuse>=5.2.0" "python-dotenv>=1.1"` (updates `pyproject.toml` `[project] dependencies` + `uv.lock`, installs into `.venv`).
- GOTCHA: `langfuse` v5 needs server ≥ 4.59.0 — running server 4.79.0 ✓. Do not pin `<5`.
- GOTCHA: do NOT add `anthropic` (project uses Ollama).
- VALIDATE: `uv run --no-sync python -c "import importlib.metadata as m; print(m.version('langfuse'), m.version('python-dotenv'))"` → `5.2.0 …`.
- MEETS: `import langfuse` works in project venv.

### CREATE `traces.py`

- DO: create `/home/jack-willis/Jack/Code/lp/p2/traces.py` with this structure (flat root module, matching sibling modules):
  - Module docstring: one line — `"""LangFuse-observed RAG ask: hybrid RRF retrieve -> cross-encoder rerank -> qwen3.5:4b grounded answer."""`
  - Env before SDK: `from dotenv import load_dotenv` then `load_dotenv()` call, THEN `from langfuse import get_client, observe` — with `# noqa: E402` on post-call import lines (LangFuse skill "Common Mistakes": import Langfuse AFTER env vars loaded; `.env` holds the keys, `uv run` does not load it).
  - Module singletons mirroring pseudocode:

    ```python
    langfuse = get_client()
    client = ollama.Client(host=OLLAMA_HOST)   # replaces anthropic.Anthropic()
    MODEL = CONTEXT_MODEL                       # qwen3.5:4b — reuse pipeline's constant, no new string
    _conn = None            # module-lazy psycopg connection (setup.connect)
    _cache: dict = {}       # shared BM25/embedding corpus cache (retrievers pattern)
    ```

  - `_session()` private helper (NOT observed): on first call `connect()` → `ensure_table(conn)` → `generate_contexts(conn)` → `backfill_embeddings(conn)` (same sequence as main.py ragas_command:33-36; all idempotent/cached), then returns `conn`. Subsequent calls return cached `_conn`.
  - `retrieve` — `@observe(as_type="retriever")`, `def retrieve(query: str, k: int = CANDIDATES) -> list[tuple[int, float]]`:

    ```python
    ranked = _rrf_ranked(_session(), PIPELINE_TABLE, _cache, body_expr=BODY_EXPR)
    hits = ranked(query, k)
    langfuse.update_current_span(
        input={"query": query, "k": k},
        output=[{"id": id, "score": score} for id, score in hits[:10]],
        metadata={"k": k, "n_hits": len(hits)},
    )
    return hits
    ```

    (k default = `CANDIDATES` = 50, matching pseudocode's k=50; `input=` set explicitly per skill best-practice "not all function args".)
  - `rerank` — `@observe()`, `def rerank(query: str, hits: list[tuple[int, float]], top_n: int = K) -> list[tuple[int, str]]`:

    ```python
    ids, bodies = _bm25_corpus(_session(), PIPELINE_TABLE, _cache, BODY_EXPR)
    body_of = dict(zip(ids, bodies))
    candidates = [(id, body_of[id]) for id, _ in hits]
    top_ids = _cross_encoder_rerank(_get_cross_encoder(), query, candidates, top_n)
    top = [(id, body_of[id]) for id in top_ids]
    langfuse.update_current_span(
        input={"query": query, "n_candidates": len(hits), "top_n": top_n},
        output=[id for id, _ in top],
    )
    return top
    ```

    (returns `(id, body)` pairs so `generate` can build the prompt without a second DB lookup.)
  - `generate` — `@observe(as_type="generation")`, `def generate(query: str, chunks: list[tuple[int, str]]) -> str`:

    ```python
    messages = [{"role": "user", "content": ANSWER_PROMPT.format(context="\n\n".join(b for _, b in chunks), question=query)}]
    langfuse.update_current_generation(
        input=messages, model=MODEL,
        model_parameters={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    )
    resp = client.chat(model=MODEL, messages=messages,
                       options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}, think=False)
    text = resp.message.content.strip()
    langfuse.update_current_generation(
        output=text,
        usage_details={"input": resp.prompt_eval_count or 0, "output": resp.eval_count or 0},
    )
    return text
    ```

    (prompt = ragas.ANSWER_PROMPT + context join ragas.py:73; ollama call = ragas._chat pattern; usage from ollama `prompt_eval_count`/`eval_count` replaces anthropic `resp.usage.input_tokens`.)
  - `ask` — `@observe(name="rag-ask")`, `def ask(query: str) -> str:` body exactly `return generate(query, rerank(query, retrieve(query)))`. Nesting (retrieve → rerank → generate under rag-ask) is automatic via `@observe`.
  - Imports (all real symbols): `from pipeline import BODY_EXPR, CONTEXT_MODEL, OLLAMA_HOST, PIPELINE_TABLE, backfill_embeddings, ensure_table, generate_contexts`; `from ragas import ANSWER_PROMPT, NUM_CTX, NUM_PREDICT`; `from retrievers import CANDIDATES, K, _bm25_corpus, _cross_encoder_rerank, _get_cross_encoder, _rrf_ranked`; `from setup import connect`; `import ollama`.
- PATTERN: retriever wiring = retrievers.py:159-175 (`rrf_ce_retriever` internals, split into the two traced steps); ollama call = ragas.py:47-57; corpus cache = ragas.py:70-72.
- GOTCHA: hits are `(id, score)` tuples — no `.id`/`.score` attributes (pseudocode adapted to unpacking).
- GOTCHA: `import langfuse` before `load_dotenv()` would read missing env → `get_client()` auth fails silently (SDK "cannot break your application" — errors only logged). Keep the dotenv-first order.
- GOTCHA: first run may trigger expensive `generate_contexts` (LLM per NULL-context chunk; cached in DB after; report `.pi/reports/rrf-ce-pipeline-report.md` says it was blocked pending ollama env flags `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`). If contexts pending, ollama must be up with `qwen3.5:4b` pulled.
- VALIDATE: `uv run --no-sync python -c "import traces; print('ok')"` → `ok`.
- MEETS: four `@observe` functions exist with pseudocode's names/signatures adapted to real project APIs; module importable.

### ADD `trace` command in `main.py`

- DO: after `ragas_command` (main.py:45), add:

  ```python
  @app.command(name="trace")
  def trace_command(question: str) -> None:
      """One LangFuse-observed RAG ask: markdown_context_chunks + RRF retrieve +
      cross-encoder rerank + qwen3.5:4b grounded answer; flushes the trace to
      Langfuse (localhost:3000)."""
      answer = ask(question)
      typer.echo(answer)
      get_client().flush()
  ```

  with top imports: `from langfuse import get_client` and `from traces import ask`.
- PATTERN: command shape = main.py:29-45 (`ragas_command`); echo = `typer.echo`.
- GOTCHA: `langfuse.flush()` is required in short-lived processes (skill "Common Mistakes": "No flush() in scripts → traces never sent"). Flush in the command, not inside `ask()` (keep `ask` matching pseudocode).
- GOTCHA: import order — `from traces import ask` runs traces' `load_dotenv()` before `from langfuse import get_client` is used at runtime; put traces import after stdlib/typer imports like sibling module imports.
- VALIDATE: `uv run --no-sync main.py trace --help` prints usage.
- MEETS: `uv run main.py trace "<question>"` prints a grounded answer and posts exactly one `rag-ask` trace.

### UPDATE `README.md` (command reference)

- DO: rewrite the empty README with a short project intro (RAG retrieval pipeline over 2 ingested books: chunkers, 5 retriever evals, contextual-retrieval pipeline with qwen3.5:4b, RAGAS faithfulness, query variants, LangFuse tracing) + prerequisites (uv, Postgres with `p2` DB, `ollama serve` with `qwen3.5:4b` pulled, LangFuse Docker stack on :3000 + `.env` keys) + one section per CLI command with what it does and what it writes:
  - `uv run main.py setup` — create 3 chunker tables, ingest both books, build GIN/HNSW indexes.
  - `uv run main.py eval` — compare 5 retrievers (ts_rank_cd, BM25, semantic, RRF, RRF+cross-encoder) across chunkers; writes 15-row `evals.md`.
  - `uv run main.py pipeline` — evaluate contextual-retrieval pipeline (markdown + qwen3.5:4b contexts + BM25 + semantic + RRF + cross-encoder rerank); idempotent, contexts cached; writes `pipeline.md`.
  - `uv run main.py ragas` — RAGAS-style faithfulness: 5 hard-coded questions through the contextual-retrieval pipeline; prints per-question scores + mean.
  - `uv run main.py variants` — query-transform eval (rewrite / multi-query / HyDE vs rrf_ce baseline); LLM transforms via local qwen3.5:4b; appends `## Query variants` to `evals.md`.
  - `uv run main.py trace "<question>"` — one LangFuse-observed RAG ask (RRF retrieve → cross-encoder rerank → qwen3.5:4b grounded answer); prints answer, posts `rag-ask` trace to Langfuse at localhost:3000.
- DO: command descriptions = `main.py` docstrings verbatim (trace = its docstring); note per-command outputs (`evals.md`, `pipeline.md`, stdout).
- PATTERN: plain markdown, `##` per section, code fences for commands; keep short — README is command reference, not design doc.
- VALIDATE: each command in `main.py` (`grep '@app.command' -A1 main.py`) appears in README; `trace` section present.
- MEETS: README documents all 6 commands including new `trace`.

### UPDATE `.gitignore`

- DO: append:

  ```
  # Secrets + upstream reference clone
  .env
  langfuse/
  ```

- GOTCHA: repo has zero commits; `.env` (live API keys) and `langfuse/` (6k-file upstream clone) would otherwise land in the first commit.
- VALIDATE: `git status --porcelain | grep -E "\.env|langfuse/"` → empty.
- MEETS: keys never committed.

## Tests

No pytest in project (convention: inline `uv run` validation). Coverage via validation ladder below plus one manual e2e. Edge cases exercised: empty retrieval (guarded inside `_rrf_ranked` fallbacks — retrievers.py:141-143), `None` token counts (`or 0` guard), first-run context generation (idempotent cached).

## Validation (real commands)

- L0 skill: `head -12 .pi/skills/langfuse/SKILL.md` · `ls .pi/skills/langfuse/references/ | wc -l` → 13
- L1 deps: `uv add "langfuse>=5.2.0" "python-dotenv>=1.1"` then `uv run --no-sync python -c "import importlib.metadata as m; print(m.version('langfuse'))"` → 5.2.0
- L2 auth (env + server reachable): `uv run --no-sync python -c "from langfuse import get_client; print(get_client().auth_check())"` → True
- L3 import smoke: `uv run --no-sync python -c "import traces"` → ok
- L4 CLI: `uv run --no-sync main.py trace --help`
- L5 manual e2e (prereqs: docker langfuse stack up ✓, `ollama serve` + `qwen3.5:4b` pulled, Postgres `p2` DB ingested):

  ```bash
  uv run main.py trace "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
  ```

  → prints answer. Then in Langfuse UI (<http://localhost:3000>): Traces → one `rag-ask` trace containing observations `retrieve` (type retriever, output = 10 id/score dicts, metadata k/n_hits), `rerank` (type span, output = top-5 chunk ids), `generate` (type generation, model qwen3.5:4b, model_parameters, usage input/output token counts). Cross-check via API:

  ```bash
  curl -s -u "$(grep LANGFUSE_PUBLIC_KEY .env | cut -d'"' -f2):$(grep LANGFUSE_SECRET_KEY .env | cut -d'"' -f2)" \
    "http://localhost:3000/api/public/v2/observations?limit=5&orderBy=timestamp" | python3 -m json.tool | grep -E '"name"|"type"'
  ```

  → shows `rag-ask`, `retrieve` (retriever), `rerank` (span), `generate` (generation).

## Acceptance

- [ ] `.pi/skills/langfuse/SKILL.md` exists with pi frontmatter (name + description, no `allowed-tools`) and 13 reference files intact.
- [ ] `traces.py` at repo root with `retrieve` (retriever), `rerank` (span), `generate` (generation, ollama qwen3.5:4b via `CONTEXT_MODEL`), `ask` (rag-ask) — all `@observe`, wired to real `_rrf_ranked`/`_cross_encoder_rerank`/`ANSWER_PROMPT`.
- [ ] `uv add` added only `langfuse` + `python-dotenv`; no `anthropic`.
- [ ] `uv run main.py trace "<golden question>"` → answer printed; trace visible on localhost:3000 with correct observation types, model name, model params, token usage.
- [ ] `.gitignore` covers `.env` and `langfuse/`.
- [ ] `README.md` documents all 6 CLI commands (setup, eval, pipeline, ragas, variants, trace) with descriptions matching `main.py` docstrings.
- [ ] All validation green; no regression in existing commands (`uv run main.py --help`).

## Open / assumed

- ASSUMED — `trace` CLI command wanted (task says verify by "run ask(...)"; CLI is the project's entry convention). If not wanted, delete task 4 and run `uv run python -c "from traces import ask; from langfuse import get_client; print(ask('...')); get_client().flush()"` instead.
- ASSUMED — python-dotenv acceptable (repo loads nothing from `.env` today; alternative = `uv run --env-file .env` on every command, keeps deps cleaner but easy to forget).
- ASSUMED — upstream skill copied verbatim (body + references) is the right adaptation; alternative = hand-written slim SKILL.md covering only Python tracing.
- ASSUMED — `markdown_context_chunks` contexts/embeddings already backfilled (last report said generation was blocked pending user-side ollama flags); if not, first `trace` run pays the one-time LLM context-generation cost.
- ASSUMED — rerank scores all retrieved hits (k=50 → CE over 50 pairs) per pseudocode, vs project's internal `RRF_CE_CANDIDATES=10` pool; kept pseudocode semantics (top_n=5 out of 50).
- Q: want `ask()` also wired into `ragas_command`/eval loop later, or CLI `trace` standalone? (Plan: standalone.)

## Notes

Rejected: `anthropic` client per pseudocode verbatim — violates "use whatever model P2 uses"; ollama is the project's only LLM path (main.py/pipeline/ragas/variants all use it). Rejected: manual trace construction via `start_observation` — `@observe` decorator matches pseudocode and auto-nests. Rejected: langfuse SDK v3/v4 pin — skill principle "use latest" + server 4.79.0 supports v5. Sequencing risk: none between tasks 1 and 2-4 (independent). Risk: first-run LLM context generation latency (mitigated: DB-cached, idempotent).

## Amendments
