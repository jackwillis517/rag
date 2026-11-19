# Plan: `ragas` command — RAGAS-style faithfulness score (ollama)

Implements: user brief (RAGAS faithfulness via new `ragas` command) | Builds on: `pipeline.py` contextual-retrieval stack | Type: enhance | Complexity: M | Systems: ollama (qwen3.5:4b), Postgres (`markdown_context_chunks`), Typer CLI, Python 3.12/uv

Grill skipped: brief delegates every open fork with "pick one / decide and justify" (verification prompt variant, empty-statement fallback, question set). The one thing the brief assumes — that `--pipeline` yields answers — is false (verified below); resolution is forced by the constraints, recorded as ASSUMED A1.

## Goal

As dev, I want `uv run python main.py ragas` to compute a RAGAS-style reference-free faithfulness score over the existing RAG stack: 5 hard-coded questions, each answered by qwen3.5:4b from the top-5 retrieved `markdown_context_chunks`; answer decomposed into statements; each statement verified supported/not-supported against the same retrieved context; per-question F = supported/total; mean over 5 printed from `main.py` — so generation groundedness is measurable locally, no external APIs.

Problem: repo has retrieval metrics only (recall/MRR/MAP — `pipeline.run()`, pipeline.py:234-266); no answer generation, no faithfulness metric. Solution: new `ragas.py` module (prompts, parsing, `faithfulness()`), new `ragas` CLI command in `main.py` that loops the questions and averages.

## Out of scope

- Not: the two relevancy metrics (brief: faithfulness only)
- Not: writing a ragas.md report file (brief: print from main.py)
- Not: touching pipeline.py / retrievers.py (pure import reuse)
- Not: pytest (project convention: inline `uv run python -c` / heredoc validation)
- Not: new deps (ollama, httpx already in uv.lock) — no RAGAS library (hand-implement per brief's notes, ~40 lines vs heavy deps)
- Not: re-chunking or any other table — context comes from `markdown_context_chunks` as-is

## Verified facts (grounded)

- **`--pipeline` produces NO answers.** `pipeline.run()` (pipeline.py:234-266) retrieves and scores only. No LLM answer step exists anywhere in the repo → `ragas` must add answer generation (A1).
- DB live (psql): `markdown_context_chunks` = 250 rows, book `Google Cloud Run`, chunk_index 0-249; context 250/250, embedding 250/250 — fully cached, so `generate_contexts` (pipeline.py:148) and `backfill_embeddings` (pipeline.py:193) are fast no-ops on this DB.
- Golden set: exactly ids 1-11 have snippet matches inside the subset (verified via `eval.relevant_ids` over `markdown_context_chunks`; matches pipeline.md "11/50 questions evaluated"). Any RAGAS question should come from that pool.
- Ollama call helper to mirror: `pipeline._one_context` (pipeline.py:124-146) — `client.chat(model=CONTEXT_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0, "num_predict": N, "num_ctx": M}, think=False)`, one retry on `(ollama.ResponseError, httpx.HTTPError)`. Client creation: `ollama.Client(host=OLLAMA_HOST)` (pipeline.py:163).
- qwen3.5:4b is a hybrid reasoner: MUST pass `think=False` (pipeline.py:139 precedent) or output contains think blocks that break regex parsing.
- Precondition gate: `generate_contexts` calls `_connectivity()` (pipeline.py:77-91) first — clean exit(1) + friendly stderr if ollama down or model not pulled. Inherited for free.
- Retrieval entry point: `rrf_ce_retriever(conn, table, cache, body_expr)` (retrievers.py:193-212) → question → top-K ids (K=5, retrievers.py:12). Contextual wiring: pipeline.py:259 `rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)`. Bodies for ids: `_bm25_corpus(conn, table, cache, body_expr)` (retrievers.py:60-69) returns `(ids, bodies)` over `coalesce(context,'') || E'\n' || content` (`BODY_EXPR`, pipeline.py:40) — same shared cache entry.
- Constants to import: `PIPELINE_TABLE` (pipeline.py:37), `CONTEXT_MODEL` (pipeline.py:32), `OLLAMA_HOST` (pipeline.py:31), `BODY_EXPR` (pipeline.py:40).
- CLI style: `@app.command(name="eval")` + `def eval_command() -> None:` + docstring (main.py:19-22). Commands are typer subcommands — brief's "--pipeline"/"--ragas" = `python main.py pipeline` / `python main.py ragas`.

## Decisions (brief-delegated, with justification)

- **D1 — verification prompt: NO-REASONING variant.** Ordered booleans are deterministic to parse (whole-text findall), fewer output tokens for a 4b model, and the brief itself notes it "is easier to parse deterministically". Tradeoff: with-reasoning variant may judge borderline statements better, but needs explanation-stripping heuristics + longer generations; swap is one constant (`VERIFY_PROMPT`).
- **D2 — brief's no-reasoning prompt lacks a context field.** It says "supported by context" but the with-reasoning variant is the one carrying `context: [context]`. Spec bug — append a `context:` block to the no-reasoning prompt.
- **D3 — empty statements -> score 1.0.** Faithfulness = groundedness: zero claims = nothing unsupported = vacuously faithful; it does NOT measure completeness (that is answer-relevancy's job). 0.0 rejected: conflates "no answer" with "lying answer". The command echoes each answer (first 80 chars) so an empty answer is never silent.
- **D4 — verdict/token mismatch -> conservative.** Parsed verdict list padded with `False` (unsupported) up to |S|; extra tokens truncated. Denominator is always |S| = len(statements).
- **D5 — the 5 questions = golden ids 1, 4, 8, 10, 11 verbatim** (verified in-subset above; spans Runtime x3, Containers, Databases). Tailored to the chunks currently in the DB by construction — no generic questions.
- **D6 — averaging lives in `main.py`** (brief constraint): `ragas_command` loops `QUESTIONS`, collects floats, computes + prints the mean. Machinery (prompts, parsing, LLM calls, retrieval) lives in `ragas.py`, matching repo layout (eval.py / pipeline.py / scorers.py flat modules).

## Create

- `ragas.py` — NEW: `QUESTIONS`, 3 prompt constants, `_chat`, `answer`, `answer_with_context`, `_statements`, `_verdicts`, `faithfulness`. ~110 lines.
- `main.py` — UPDATE: new imports + `ragas_command` (~20 lines; file currently 34 lines).

## Patterns (existing repo)

Ollama call (mirror pipeline.py:124-146):

```python
resp = client.chat(
    model=CONTEXT_MODEL,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    think=False,
)
return resp.message.content.strip()
```

CLI command (mirror main.py:19-22):

```python
@app.command(name="eval")
def eval_command() -> None:
    """..."""
    run_eval()
```

conn + shared cache (mirror pipeline.py:256-259): `cache: dict = {}` then `with connect() as conn:` — one conn, one cache for all 5 questions.

## Exact prompts (module constants in ragas.py)

```python
ANSWER_PROMPT = (
    "Answer the question using only the information in the provided context. "
    "If the context does not contain the answer, say so.\n"
    "context:\n{context}\n"
    "question: {question}"
)

EXTRACT_PROMPT = (  # brief, verbatim shape
    "Given a question and answer, create one or more statements from each sentence "
    "in the given answer.\n"
    "question: {question}\n"
    "answer: {answer}"
)

VERIFY_PROMPT = (  # brief no-reasoning variant + D2 context block
    "For each numbered statement, output true if supported by context, false otherwise. "
    "Output exactly one boolean per statement, in order.\n"
    "statements:\n{statements}\n"
    "context:\n{context}"
)
```

`{statements}` = numbered list: `"\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))`.
`{context}` = `"\n\n".join(body_of[i] for i in ids)` (top-5 combined bodies).

## Parsing

Statements (extraction output) — line-based, strip leading list markers/numbers only (anchored `^`, sentence text untouched):

```python
def _clean_line(line: str) -> str:
    line = line.strip().lstrip("-* ").strip()
    return re.sub(r"^(?:statement\s*)?\d+[.:)]\s*", "", line, flags=re.IGNORECASE).strip()
# statements = [s for s in (_clean_line(l) for l in out.splitlines()) if s]
```

Verdicts (verification output) — ordered regex findall over whole text (order-preserving; survives `1. true`, bare `true`, or inline `true false true`):

```python
_TOKEN_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
tokens = _TOKEN_RE.findall(out)
verdicts = [t.lower() in ("true", "yes") for t in tokens]
verdicts = (verdicts + [False] * n)[:n]   # pad unsupported, truncate extras (D4)
```

GOTCHA: `\b` word boundaries — `not`/`now`/`nothing` do NOT match `\bno\b` (word char follows the `no`). `Yes/No` accepted per brief.

Score: `faithfulness` = `sum(verdicts) / len(statements)`; `if not statements: return 1.0` (D3).

## Tasks

### CREATE `ragas.py`
- DO: module docstring ("RAGAS-style faithfulness: qwen3.5:4b answer grounded in retrieved chunks; statement extraction + verification; F = supported/total; reference-free").
- DO: constants:
```python
NUM_CTX = 8192      # 5 chunks (~2k tokens) + prompts; far under qwen3.5:4b max 262144 (pipeline.py:33)
NUM_PREDICT = 512   # cap; verdict/extraction outputs are shorter — max, not target
```
- DO: `QUESTIONS: list[str]` — the 5 hard-coded questions (D5, golden ids 1/4/8/10/11 verbatim):
  1. "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
  2. "What does setting concurrency to 1 mean for a Cloud Run service?"
  3. "Is the Cloud Run container filesystem backed by disk, and how large can it get?"
  4. "How does a multistage Dockerfile let you ship a small image with only the compiled binary?"
  5. "How does the Cloud SQL Proxy Server authenticate incoming connections?"
- DO: imports — `re`, `httpx`, `ollama`; `from pipeline import BODY_EXPR, CONTEXT_MODEL, OLLAMA_HOST, PIPELINE_TABLE`; `from retrievers import _bm25_corpus, rrf_ce_retriever`.
- DO: signatures (client passed in — mirrors `_one_context(client, ...)`):
```python
def _chat(client: ollama.Client, prompt: str) -> str
    # chat per Patterns; ONE retry on (ollama.ResponseError, httpx.HTTPError); temperature 0, think=False

def answer(client: ollama.Client, question: str, context: str) -> str
    # ANSWER_PROMPT.format(...) -> _chat

def answer_with_context(client: ollama.Client, conn, question: str, cache: dict) -> tuple[str, str]
    # retriever = rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)
    # ids = retriever(question)                        # top-5 (K, retrievers.py:12)
    # ids_all, bodies = _bm25_corpus(conn, PIPELINE_TABLE, cache, BODY_EXPR)
    # body_of = dict(zip(ids_all, bodies))             # same cache entry — zero extra queries
    # context = "\n\n".join(body_of[i] for i in ids)   # ids empty -> "" (defined edge)
    # return answer(client, question, context), context

def _statements(client: ollama.Client, question: str, answer: str) -> list[str]
    # EXTRACT_PROMPT -> _chat -> line parse (Parsing)

def _verdicts(client: ollama.Client, statements: list[str], context: str) -> list[bool]
    # numbered statements + VERIFY_PROMPT -> _chat -> ordered findall, pad/truncate to len(statements)

def faithfulness(client: ollama.Client, question: str, answer: str, context: str) -> float
    # statements = _statements(client, question, answer)
    # if not statements: return 1.0                    (D3)
    # verdicts = _verdicts(client, statements, context)
    # return sum(verdicts) / len(statements)
```
- PATTERN: `pipeline._one_context` pipeline.py:124-146 (chat options + retry); retriever wiring pipeline.py:259; `_bm25_corpus` retrievers.py:60-69.
- GOTCHA: MUST pass `think=False` — qwen3.5:4b emits think blocks otherwise, corrupting parses. Read `resp.message.content` (not `.text`).
- VALIDATE: `uv run python -c "import ragas; print(len(ragas.QUESTIONS))"`
- MEETS: one float score per (question, answer, context); pure-local LLM calls.

### UPDATE `main.py`
- DO: extend imports (keep existing ones; add):
```python
import ollama
from pipeline import OLLAMA_HOST, backfill_embeddings, ensure_table, generate_contexts
from ragas import QUESTIONS, answer_with_context, faithfulness
```
(keep `from pipeline import run as run_pipeline` as-is)
- DO: append command before `if __name__ == "__main__":` (main.py:33):
```python
@app.command(name="ragas")
def ragas_command() -> None:
    """RAGAS-style faithfulness: 5 hard-coded questions through the contextual-retrieval
    pipeline (qwen3.5:4b answers grounded in top-5 retrieved chunks); statements verified
    against the same context; prints per-question scores and their mean."""
    client = ollama.Client(host=OLLAMA_HOST)
    cache: dict = {}
    scores: list[float] = []
    with connect() as conn:
        ensure_table(conn)
        generate_contexts(conn)      # _connectivity gate; cached contexts -> fast
        backfill_embeddings(conn)   # cached embeddings -> fast
        for question in QUESTIONS:
            answer, context = answer_with_context(client, conn, question, cache)
            score = faithfulness(client, question, answer, context)
            scores.append(score)
            typer.echo(f"  {score:.2f}  {question}  |  answer: {answer[:80]}")
    mean = sum(scores) / len(scores)
    typer.echo(f"Mean faithfulness ({len(scores)} questions): {mean:.2f}")
```
- PATTERN: command registration main.py:19-22; client creation pipeline.py:163; conn+cache pipeline.py:256-259.
- GOTCHA: averaging literally in main.py (brief constraint, D6). First call order matters: `generate_contexts` runs `_connectivity()` (clean exit if ollama down) before any ragas LLM call.
- VALIDATE: `uv run python main.py --help`
- MEETS: `python main.py ragas` prints 5 per-question scores + mean.

## Tests

Project convention: no pytest — inline `uv run python` heredocs. Seams: `faithfulness` is end-to-end over a local ollama (temp 0); pure-logic pieces (`_clean_line`, verdict pad/truncate) exercised via crafted LLM-free cases where possible.

```bash
uv run python - <<'PY'
import ollama
from pipeline import OLLAMA_HOST
from ragas import faithfulness

client = ollama.Client(host=OLLAMA_HOST)
q = "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
ctx = ("You'll get 10 seconds to shut down. You should close database connections "
       "and flush buffers programmatically.")
print("grounded:", faithfulness(client, q, "You'll get 10 seconds to shut down cleanly.", ctx))
print("liar    :", faithfulness(client, q, "The container gets 600 seconds to shut down.", ctx))
print("empty   :", faithfulness(client, q, "", ctx))
PY
```

- grounded -> expect >= 0.75 (temp 0 but 4b verification is slightly noisy; 1.0 typical)
- liar -> expect <= 0.5 (unsupported statements dominate)
- empty -> expect exactly 1.0 (D3 branch, no LLM call needed for statements... note: still calls _statements, which returns [] -> short-circuit)
- Verdict pad edge: if a verification ever returns fewer tokens than statements, missing ones count False (D4) — observable by inspecting a per-question score < 1 with answer[:80] echo.

## Validation (real commands)

- L1 imports: `uv run python -c "import ragas, main"` (repo has no lint config — convention from rrf-ce plan)
- L2 CLI: `uv run python main.py --help` -> `ragas` listed
- L3 seams: heredoc block above (grounded / liar / empty)
- L4 full run: `uv run python main.py ragas` -> 5 per-question lines + `Mean faithfulness (5 questions): X.XX`, X in [0.00, 1.00]; ~15 sequential qwen3.5:4b calls (3 per question) — expect ~1-3 min
- L5 manual: (a) run twice — scores near-identical (temperature 0), second run fast (contexts/embeddings cached); (b) stop ollama (`pkill ollama`) then `python main.py ragas` -> exit 1 with `_connectivity` friendly message (inherited via `generate_contexts`, pipeline.py:77-91); (c) `python main.py eval` and `python main.py pipeline` still work (no edits to their modules)

## Acceptance

- [ ] `uv run python main.py ragas` prints 5 per-question faithfulness scores + mean in [0,1]
- [ ] All LLM calls via ollama qwen3.5:4b, `think=False`, temperature 0; no external APIs; no new deps (uv.lock untouched)
- [ ] `faithfulness(client, question, answer, context) -> float` computes ONE score for one (question, answer, context); averaging lives in `main.py`
- [ ] Context = combined (LLM context + content) bodies of top-5 `markdown_context_chunks` via `rrf_ce_retriever` — same retrieval as `--pipeline`, no re-chunking
- [ ] Empty-statements fallback returns 1.0 and is exercised (L3)
- [ ] L1-L5 green; `eval` / `pipeline` commands unaffected

## Open / assumed

- ASSUMED — A1: brief assumes `--pipeline` yields (answer, chunks); it yields chunks only (pipeline.py:234). Resolution forced by constraints: `ragas` generates answers itself with qwen3.5:4b grounded in the retrieved context (`ANSWER_PROMPT` above). If a different answer-gen prompt is preferred, it is one constant to swap.
- ASSUMED — A2: ollama running + `qwen3.5:4b` pulled (same precondition as `--pipeline`; enforced by `_connectivity`).
- ASSUMED — A3: `NUM_CTX = 8192` suffices — 5 chunks ~ 5 x (context ~50 + content ~800 chars) ~ 2k tokens; prompts tiny. If truncation observed, raise to 16384, nothing else changes.
- Open: question set swappable via `QUESTIONS` — keep questions inside the current DB subset (golden ids 1-11 are the verified in-subset pool; screen additions with `eval.relevant_ids`).
- Open: if no-reasoning verification proves unreliable (token count frequently != statement count), switch `VERIFY_PROMPT` to the with-reasoning variant and parse the LAST yes/no token per line instead of whole-text findall (small amendment, documented).

## Notes

- Rejected: RAGAS library — pulls datasets/langchain, violates "no new dependencies"; hand-impl is ~40 lines per the brief's own notes.
- Rejected: with-reasoning verification as default (D1) — parse fragility + token cost on a 4b model; kept as documented fallback.
- Rejected: writing ragas output to a .md file — brief says print from main.py; the evals.md/pipeline.md file-writing pattern exists if a report is wanted later.
- Rejected: per-statement verification calls (N calls instead of 1) — up to ~40 LLM calls, slower, no accuracy win at temp 0.
- Tradeoff: whole-text findall (vs per-line) is robust to inline boolean lists but can over-collect tokens if the model echoes statements containing the words true/false; temp 0 + "Output exactly one boolean per statement, in order" mitigates; padding is conservative (D4) so over-collection only truncates.
- Sequencing: single new module + one command; pipeline.py/retrievers.py import-only reuse -> no regression surface. Runtime risk: none beyond ollama availability.

## Amendments
(append-only: date - change - why)

- 2026-09-18 - `faithfulness` guards `not answer.strip()` before extraction — L3 showed qwen3.5:4b hallucinates statements from question alone when answer is empty (raw output confirmed), so `if not statements: return 1.0` never fires; guard makes D3 deterministic. Why: plan assumed `_statements("") == []`; model proved otherwise.
