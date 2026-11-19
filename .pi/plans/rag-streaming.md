# Plan: tracing-free `rag.py` (retrieve + rerank + async streaming cited answer)

Implements: user task (new `rag.py`: retrieve/rerank matching traces.py minus tracing; async generator `stream_answer(client, query, chunks)` over ollama AsyncClient) | Type: new module | Complexity: L-M | Systems: Python 3.12/uv flat modules, Postgres 16.15 + pgvector, Ollama qwen3.5:4b | Deps: none new

User-settled decisions (grill round): cited prompt (a) · async generator yielding content deltas (Q2) · `ollama.AsyncClient` param (Q3) · model/options verbatim from traces.generate (Q4) · separate module, no traces.py refactor (Q5).

## Goal

As dev, I want a tracing-free RAG module — `retrieve`, `rerank`, and an async streaming answer generator — so that non-LangFuse consumers (CLI, server, scripts) get token-by-token cited answers over the same retrieval path without importing langfuse or needing `.env` keys.

## Out of scope

- Not: `traces.py` changes of any kind (no refactor, no delegation — Q5 settled).
- Not: `ragas.py` (untouched; rag.py imports only `NUM_CTX` from it).
- Not: LangFuse anything — no `@observe`, no `langfuse.*` calls, no langfuse import (direct or transitive).
- Not: new `main.py` CLI command (see Open/assumed Q — sync-vs-async entry stays open).
- Not: pytest (project convention = inline `uv run` validation).

## Read first (required)

- `traces.py` (whole, 172 lines) — the exact code being mirrored minus tracing: `_session` (56-64), `retrieve` (68-77), `rerank` (81-95), `generate`'s metadata SELECT + sources block (98-116), `CITED_ANSWER_PROMPT` (35-52), `NUM_PREDICT_CITED` (33).
- `pipeline.py` (17-31) — `OLLAMA_HOST`, `CONTEXT_MODEL`, `PIPELINE_TABLE`, `BODY_EXPR`, `ensure_table`, `generate_contexts`, `backfill_embeddings` (all imported by traces.py:12-18).
- `ragas.py` (15, 28-33) — `NUM_CTX = 8192`; `ANSWER_PROMPT` (contrast: NOT used by rag.py).
- `retrievers.py` — `CANDIDATES`, `K`, `_rrf_ranked`, `_bm25_corpus`, `_get_cross_encoder`, `_cross_encoder_rerank`.
- `setup.py` — `connect`.
- `.pi/plans/chunk-metadata-citations.md` — implemented; source of the citation metadata columns rag.py's sources block reads.

## Facts established (repo + live DB + ollama SDK, verified this session)

- Chunk-metadata plan IS implemented: `markdown_context_chunks` has 13 columns incl. `source_file, char_start, char_end, line_start, chunk_key`; 0 NULL `source_file` (SQL verified). traces.py carries `CITED_ANSWER_PROMPT` (35-52, incl. trailing "Citations:" section spec), `NUM_PREDICT_CITED = 768` (33), metadata SELECT by `id = ANY(%s)` (101-107), sources block `[n] {book_title} — {heading_path} ({source_file}, line {line_start})\n{body}` (108-112).
- **Why rag.py defines its own copy, not `from traces import CITED_ANSWER_PROMPT`** (settled decision): traces.py at module level does `load_dotenv()` (5), `from langfuse import get_client, observe` (7), and `langfuse = get_client()` (30). Importing traces from rag.py couples a tracing-free module to the langfuse SDK and `.env` keys — defeats the module's purpose. Q5 forbids extracting a shared prompts module (traces refactor). → verbatim copy + drift guard in validation ladder.
- ollama 0.6.3 async streaming (verified in installed `ollama/_client.py` source):
  - `ollama.AsyncClient` exists; `chat` is a **coroutine**: body is `return await self._request(ChatResponse, 'POST', '/api/chat', json=ChatRequest(..., stream=stream, think=think, ...), ...)`.
  - With `stream=True`, `_request` returns `inner()` — an **async generator directly**. Correct consumption: `stream = await client.chat(..., stream=True)` then `async for part in stream:`. NOT `async for part in client.chat(...)` (chat is a coroutine, not an async gen) and NOT `await` twice.
  - Each part is a `ChatResponse`; delta text at `part.message.content` (may be `None` on empty/done parts → `or ""`). `think=False` supported kwarg (verified in signature) — no thinking tokens in the stream, matches traces.generate.
  - Server down → `ollama.ConnectionError` (from `_request_raw`, catches `httpx.ConnectError`); bad request → `ollama.ResponseError`; both raise from inside the generator to the consumer.
- Repo retry convention: single retry on transient `(ollama.ResponseError, httpx.HTTPError)` (ragas.py:47-57 `_chat`, pipeline.py:118-131 `_one_context`). Applies to full-response calls only — a partially-consumed stream cannot be retried without duplicating output. Connectivity pre-check convention: `pipeline._connectivity()` (typer.Exit with human message).
- Live retrieval path is deterministic: RRF fusion + cross-encoder (`ms-marco` CE, inference only, no temperature) → rag.retrieve and traces.retrieve produce identical output for same query (used as a validation assert).
- `main.py` has no async entry today; Typer commands are sync functions.

## Create

- `rag.py` — repo-root flat module: `_session`, `retrieve`, `rerank`, `_sources_block`, `stream_answer`. No langfuse anywhere.

## Patterns (real excerpts from this repo)

traces.py:68-77 — retrieve to mirror exactly (minus `@observe` + `langfuse.update_current_span`):

```python
def retrieve(query: str, k: int = CANDIDATES) -> list[tuple[int, float]]:
    """Hybrid RRF retrieval (semantic + BM25 fusion) over the pipeline table."""
    ranked = _rrf_ranked(_session(), PIPELINE_TABLE, _cache, body_expr=BODY_EXPR)
    hits = ranked(query, k)
    ...
    return hits
```

traces.py:98-116 — metadata SELECT + sources block to mirror (the only part of `generate` that `stream_answer` keeps; prompt build + chat call become streaming):

```python
ids = [id for id, _ in chunks]
with _session().cursor() as cur:
    cur.execute(
        f"""SELECT id, book_title, heading_path, source_file, char_start, char_end, line_start, chunk_key
            FROM {PIPELINE_TABLE} WHERE id = ANY(%s)""",
        (ids,),
    )
    meta = {row[0]: row for row in cur.fetchall()}
sources = "\n\n".join(
    f"[{n}] {meta[id][1]} — {meta[id][2] or ''} ({meta[id][3]}, line {meta[id][6]})\n{body}"
    for n, (id, body) in enumerate(chunks, 1)
)
```

traces.py:144-152 — options dict to reuse verbatim in the streaming chat call:

```python
options={
    "temperature": 0,
    "num_predict": NUM_PREDICT_CITED,
    "num_ctx": NUM_CTX,
},
think=False,
```

## Tasks

Single task — file is small and self-contained; sub-steps within it are the build order.

### CREATE `rag.py`

- DO: create `/home/jack-willis/Jack/Code/lp/p2/rag.py`, flat root module. Structure:

  1. Module docstring: `"""Tracing-free RAG ask: hybrid RRF retrieve -> cross-encoder rerank -> async streaming cited answer (qwen3.5:4b)."""` (mirror traces.py:1, minus the LangFuse mention).
  2. Imports (alphabetical, sibling style — traces.py:3-31 minus dotenv/langfuse, plus `collections.abc.AsyncIterator`):

     ```python
     from collections.abc import AsyncIterator

     import ollama
     from pipeline import (
         BODY_EXPR,
         CONTEXT_MODEL,
         PIPELINE_TABLE,
         backfill_embeddings,
         ensure_table,
         generate_contexts,
     )
     from ragas import NUM_CTX
     from retrievers import (
         CANDIDATES,
         K,
         _bm25_corpus,
         _cross_encoder_rerank,
         _get_cross_encoder,
         _rrf_ranked,
     )
     from setup import connect
     ```

     Note: `OLLAMA_HOST` NOT imported — `stream_answer` takes the client as a parameter (user spec); the module never constructs a client.
  3. Constants:

     ```python
     MODEL = CONTEXT_MODEL  # qwen3.5:4b
     NUM_PREDICT_CITED = 768  # cited answers run longer than ragas' 512 cap (mirror traces.py:33)
     ```

  4. `CITED_ANSWER_PROMPT` — **verbatim byte-for-byte copy of traces.py:35-52** including the "Citations:" trailing-section spec and `{sources}` / `{query}` placeholders. Add a one-line comment: `# verbatim copy of traces.CITED_ANSWER_PROMPT — kept separate so rag.py imports no langfuse; keep in sync (validation ladder asserts equality)`.
  5. Module state + `_session()` — exact copy of traces.py:54-64 (`_conn = None`, `_cache: dict = {}`, lazy `connect()` → `ensure_table` → `generate_contexts` → `backfill_embeddings`). Comment: mirrors traces.py, no tracing.
  6. `retrieve` — exact copy of traces.py:68-77 minus `@observe(as_type="retriever")` decorator and the whole `langfuse.update_current_span(...)` block. Signature, docstring, body, return unchanged.
  7. `rerank` — exact copy of traces.py:81-95 minus `@observe()` and `langfuse.update_current_span(...)`. Returns `list[tuple[int, str]]` `(id, body)` pairs, body = `BODY_EXPR`.
  8. `_sources_block(chunks: list[tuple[int, str]]) -> str` — private helper = traces.py:100-113 verbatim (metadata SELECT by `id = ANY(%s)` over `_session()`, `meta` dict, numbered sources join). Extracted as helper so `stream_answer` stays one screen.
  9. Async generator:

     ```python
     async def stream_answer(
         client: ollama.AsyncClient, query: str, chunks: list[tuple[int, str]]
     ) -> AsyncIterator[str]:
         """Async streaming cited answer; yields content deltas.

         Consumed as:
             stream = await client-init happens inside chat coroutine...
         """
         sources = _sources_block(chunks)
         messages = [
             {
                 "role": "user",
                 "content": CITED_ANSWER_PROMPT.format(sources=sources, query=query),
             }
         ]
         stream = await client.chat(
             model=MODEL,
             messages=messages,
             options={
                 "temperature": 0,
                 "num_predict": NUM_PREDICT_CITED,
                 "num_ctx": NUM_CTX,
             },
             stream=True,
             think=False,
         )
         async for part in stream:
             yield part.message.content or ""
     ```

- PATTERN: `retrieve`/`rerank` = traces.py:68-95 (verbatim, tracing lines deleted); `_sources_block` = traces.py:100-113; options/model = traces.py:135-152 constants (`MODEL = CONTEXT_MODEL`, `NUM_PREDICT_CITED`, `NUM_CTX`); await-then-async-for = verified ollama 0.6.3 `_client.py` (`return await self._request(...)` → `_request` stream path `return inner()`).
- GOTCHA: **`await client.chat(...)` is required** — `AsyncClient.chat` is a coroutine returning the async generator; `async for part in client.chat(...)` raises `TypeError` (coroutine not iterable). And do NOT `await` the returned object — it's already the async generator.
- GOTCHA: `part.message.content` can be `None` (empty deltas / final done part) → `or ""` keeps every yield a `str`, consumer concatenation safe.
- GOTCHA: async generator body is lazy — `_sources_block`'s DB SELECT runs on the consumer's FIRST `__anext__`, not at call time. First consumer iteration may pay one-time `_session()` init (ensure_table/contexts/embeddings — all cached afterwards).
- GOTCHA: no retry inside the generator (repo retry convention = ragas.py:47-57, but a partially-consumed stream cannot be retried without duplicating already-yielded output; acquisition errors happen pre-first-yield, mid-stream errors propagate). Consumer owns error handling (see Tests).
- GOTCHA: prompt copy drift — traces.py:35-52 is the single source of truth for now; if either copy changes the other silently diverges. Validation ladder asserts `rag.CITED_ANSWER_PROMPT == traces.CITED_ANSWER_PROMPT` (test-only import of traces is fine — langfuse + `.env` present in dev env).
- GOTCHA: `id` shadows builtin — matches repo style (traces.py:70, ragas.py:72); keep for verbatim mirroring.
- GOTCHA: empty `chunks` list → `_sources_block` yields `""` sources → model told to say sources don't contain the answer. Harmless; assert covered in Tests.
- VALIDATE:

  ```bash
  uv run --no-sync python -c "import rag; print('import ok')"
  uv run --no-sync python -c "
  import sys, rag
  assert not any(m.split('.')[0] == 'langfuse' for m in sys.modules), 'langfuse leaked'
  assert not hasattr(rag, 'langfuse') and not hasattr(rag, 'observe')
  print('tracing-free: ok')"
  uv run --no-sync python -c "
  import rag, traces
  assert rag.CITED_ANSWER_PROMPT == traces.CITED_ANSWER_PROMPT, 'prompt drift'
  assert rag.NUM_PREDICT_CITED == traces.NUM_PREDICT_CITED
  assert rag.MODEL == traces.MODEL
  print('constants in sync')"
  uv run --no-sync python -c "
  import inspect, rag
  src = inspect.getsource(rag)
  for fn in ('retrieve', 'rerank'):
    t = inspect.getsource(getattr(traces if False else __import__('traces'), fn))
  import traces as tr
  for fn in ('retrieve', 'rerank'):
    a = inspect.getsource(getattr(rag, fn)); b = inspect.getsource(getattr(tr := tr, fn))
    # strip decorators: compare from first 'def' line, and drop langfuse lines
    norm = lambda s: [l for l in s.splitlines() if 'langfuse' not in l and 'observe' not in l and not l.strip().startswith('@')][1:]
    assert norm(a) == norm(b), fn
  print('retrieve/rerank match traces minus tracing')"
  ```

  Last check imports traces (fine — validation env has `.env` + langfuse server); asserts rag's function bodies equal traces' modulo decorator/langfuse lines.
- MEETS: `rag.py` exists at repo root; retrieve/rerank byte-equivalent to traces modulo tracing; async streaming generator consumes ollama AsyncClient per verified 0.6.3 API; zero langfuse coupling.

### Consumer contract (documented, not coded)

- DO: nothing in source — this is the invocation contract consumers follow:
  - **Invocation** (how a consumer calls it):

    ```python
    import asyncio
    import ollama
    from pipeline import OLLAMA_HOST
    from rag import retrieve, rerank, stream_answer

    async def main() -> None:
        chunks = rerank(q, retrieve(q))          # sync, tracing-free
        client = ollama.AsyncClient(host=OLLAMA_HOST)
        async for delta in stream_answer(client, q, chunks):
            print(delta, end="", flush=True)

    asyncio.run(main())
    ```

    (`chunks` may be computed outside the event loop — retrieve/rerank are sync and heavy (CE model load); compute before creating/streaming tasks.)
  - **Error handling per repo conventions**:
    - Server down: `ollama.ConnectionError` (httpx.ConnectError wrapped, human message) raised at first iteration. Pre-check with the `pipeline._connectivity()` pattern when a friendly message matters (pipeline.py:60-73: `ollama.list()` probe → human hint). `pipeline._connectivity` itself is sync/typer-bound — mirror its probe inline in consumers that need it.
    - Mid-stream failure: `ollama.ResponseError` / `httpx.HTTPError` propagate out of the `async for` — no retry, per the no-partial-retry rule above. Consumers needing resilience collect deltas, catch, and restart the whole `stream_answer` call (fresh generator), discarding partial output.
    - Truncation: `num_predict = 768` — stream simply ends; final part carries `done_reason`. Consumers checking completeness inspect the last part before the generator exhausts (or accept truncation silently, as traces.generate does).
- MEETS: any stranger can wire rag.py into a CLI/server without reading ollama docs.

## Tests

No pytest (convention: inline `uv run`). Coverage:
- Import purity: no langfuse modules loaded after `import rag` (validation block above).
- Equivalence: `rag.retrieve(q) == traces.retrieve(q)` and `rag.rerank(q, ...) == traces.rerank(q, ...)` on one golden query — deterministic RRF+CE path (validation block + e2e below).
- Prompt/constant parity: equality asserts vs traces (drift guard).
- E2E streaming (L4 below): deltas non-empty, concatenation contains `[1]` citation and ends with `Citations:` section; delta count > 1 proves streaming.
- Edge: empty `chunks` → model answers "sources don't contain the answer"; no KeyError (`_sources_block` builds sources from chunks list, SELECT with empty `ANY(%s)` returns no rows — but the join iterates `chunks`, not `meta`, so no id lookup happens; keep that ordering — GOTCHA: traces.generate builds from `chunks` enumeration too, same safety).
- Edge: `part.message.content is None` → yields `""`, never crashes consumer concat.

## Validation (real commands)

Prereqs: Postgres `p2` ingested (markdown_context_chunks 250 rows, metadata backfilled — verified live), `ollama serve` up + `qwen3.5:4b` pulled.

- L1 imports: `uv run --no-sync python -c "import rag; print('ok')"`.
- L2 purity: langfuse-free import check (task VALIDATE block) → `tracing-free: ok`.
- L3 parity: prompt/constant/retrieve/rerank equivalence blocks (task VALIDATE) → `constants in sync`, `retrieve/rerank match traces minus tracing`, plus live equivalence:

  ```bash
  uv run python -c "
  from rag import retrieve, rerank
  from traces import retrieve as tr, rerank as rr
  q = 'How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?'
  hits = retrieve(q); assert hits == tr(q)
  top = rerank(q, hits); assert top == rr(q, tr(q)); print('equivalence ok', [i for i,_ in top])
  "
  ```

  (run once; traces call posts LangFuse spans — expected, harmless).
- L4 e2e streaming:

  ```bash
  uv run python -c "
  import asyncio, ollama
  from pipeline import OLLAMA_HOST
  from rag import retrieve, rerank, stream_answer
  q = 'How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?'
  async def main():
      chunks = rerank(q, retrieve(q))
      client = ollama.AsyncClient(host=OLLAMA_HOST)
      parts = [d async for d in stream_answer(client, q, chunks)]
      assert len(parts) > 1, 'not streaming'
      text = ''.join(parts)
      assert '[1]' in text, 'no citation'
      assert 'Citations:' in text, 'no citations section'
      print(f'streamed {len(parts)} deltas, {len(text)} chars'); print(text)
  asyncio.run(main())
  "
  ```

  → answer prints with per-claim `[n]` citations + trailing `Citations:` section (SIGTERM answer expected ~ "10 seconds" grace period + citations).
- L5 no-regression: `uv run --no-sync main.py --help` → all commands list; `uv run --no-sync python -c "import traces, ragas, main"` → ok (traces untouched).

## Acceptance

- [ ] `rag.py` at repo root; imports no langfuse (direct or transitive) — L2 green.
- [ ] `retrieve(query, k=CANDIDATES) -> list[tuple[int, float]]` and `rerank(query, hits, top_n=K) -> list[tuple[int, str]]` byte-equivalent to traces.py:68-95 modulo `@observe`/`langfuse.update_*` — L3 green on live query.
- [ ] `stream_answer(client: ollama.AsyncClient, query, chunks) -> AsyncIterator[str]` async generator; `await client.chat(..., stream=True)` + `async for` pattern per verified ollama 0.6.3 API; options = temperature 0, `num_predict 768` (`NUM_PREDICT_CITED`), `num_ctx 8192` (`ragas.NUM_CTX`), `think=False`; model = `CONTEXT_MODEL`.
- [ ] L4 e2e: >1 deltas, `[1]` in output, `Citations:` section present.
- [ ] `rag.CITED_ANSWER_PROMPT == traces.CITED_ANSWER_PROMPT` and `NUM_PREDICT_CITED`/`MODEL` parity asserts green.
- [ ] `traces.py`, `ragas.py`, `main.py` untouched (git diff empty).
- [ ] No new dependencies.

## Open / assumed

- ASSUMED — rag.py defines a **verbatim copy** of `CITED_ANSWER_PROMPT` + `NUM_PREDICT_CITED` (768) rather than importing from traces: traces.py module-level `get_client()` + `.env` requirement would couple the tracing-free module to langfuse; Q5 forbids the shared-module extraction. Drift risk handled by L3 parity assert. If drift ever bites: promote both prompts to a new `prompts.py` in a later task (traces refactor — currently out of scope).
- ASSUMED — `_session()` module-lazy init copied from traces: first `retrieve`/`stream_answer` call may pay one-time ensure_table/contexts/embeddings (all cached in DB after). Documented in docstring.
- ASSUMED — no retry inside `stream_answer` (partial-stream retry unsafe); consumers own resilience. Deviates from ragas.py:47-57 retry convention deliberately.
- Q (open — user call): CLI entry. `main.py` commands are sync; a streaming `ask` command would need `asyncio.run` (Typer sync wrapper) or anyio. Plan: **no CLI task** — rag.py ships library-only. If wanted, tiny follow-up task: `@app.command() def ask(query: str)` body `asyncio.run(_stream_to_stdout(query))` with `print(delta, end="", flush=True)`; decide later.
- Q (open — user call): should the final part's usage (`prompt_eval_count`/`eval_count`, present on the done part in stream mode) be surfaced (e.g. yielded as a sentinel or a second return channel)? Plan: no — deltas only, matches user spec ("yielding content deltas").

## Notes

Rejected: `from traces import CITED_ANSWER_PROMPT` — drags langfuse SDK + `.env` into a tracing-free module (traces.py:5,7,30 module-level side effects). Rejected: extracting shared `prompts.py` — Q5 settled "no traces.py refactor"; revisit if a third consumer appears. Rejected: sync generator wrapping a thread — ollama ships a first-class AsyncClient (verified in 0.6.3 source); threads would fake what httpx.AsyncClient already provides. Rejected: retry inside the generator (ragas convention) — duplicating already-yielded deltas on mid-stream retry is worse than propagating; acquisition-only retry saves one failure mode but complicates the two-step await pattern for marginal gain. Rejected: passing `conn` into retrieve/rerank — breaks "match traces.py exactly" (module-lazy `_session()` is the traces shape). Tradeoff: prompt duplication vs import purity — purity wins per the module's stated purpose; parity assert is the drift tripwire. Sequencing risk: none — single new file, zero edits to existing code.

## Amendments
