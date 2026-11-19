# Plan: `variants` command — query rewriting + multi-query + HyDE vs rrf_ce baseline

Implements: user brief (3 LLM query transforms evaluated on 10 queries) | Builds on: `pipeline.py` contextual stack + implemented `.pi/plans/ragas-faithfulness.md` (its `ragas._chat` is the LLM primitive) | Type: enhance | Complexity: M | Systems: ollama (qwen3.5:4b), Postgres (`markdown_context_chunks`), sentence-transformers (bi-encoder + cross-encoder), Typer CLI, Python 3.12/uv

Grill: done, 6 questions answered by user — Q1 ids 1-10 verbatim; Q2 `markdown_context_chunks` ONLY (not 3 tables); Q3 per-variant RRF top-10 → union → CE anchored on ORIGINAL question; Q4 baseline row + separate evals.md section; Q5 HyDE replaces semantic-leg embedding ONLY; Q6 temp 0, recompute each run, no cache. One un-grilled fork → D7 below (ASSUMED).

## Goal

As dev, I want `uv run python main.py variants` to evaluate 3 LLM query-transform retrievers — query rewriting, multi-query retrieval, HyDE — against a plain `rrf_ce` baseline, all four with the base pipeline matching the rrf-ce configuration (contextual `markdown_context_chunks` corpus, BM25 + semantic RRF fusion, cross-encoder rerank, k=5), on the SAME 10 golden questions, computing recall@5 / MRR / MAP for each, so query-transformation value is measurable against the existing stack.

Problem: repo has zero query-time LLM transforms — every retriever in `eval.py` embeds/ranks the raw question (`retrievers.py:48` ts, `:91` BM25, `:116` semantic, `:147` RRF, `:193` rrf_ce). Solution: new `variants.py` (prompts, parsers, 3 retriever factories, `run()` harness writing a `## Query variants` section into evals.md) + new `variants` CLI command + a section-preserve tweak in `eval.py`.

## Out of scope

- Not: the two RAGAS relevancy metrics (separate brief)
- Not: running the 3 new retrievers over `fixed_chunks`/`recursive_chunks` (settled Q2: contextual table only)
- Not: changing `eval` metrics/retrievers/K — the 15-row section must stay byte-identical (D8)
- Not: rewriting `retrievers.py` — import its helpers, zero edits (no regression surface)
- Not: pytest (project convention: inline `uv run python` validation)
- Not: new deps — ollama/httpx/ranx already in uv.lock
- Not: LLM transform caching (settled Q6: recomputed each run, ~30 short calls)
- Not: HyDE doc text used by BM25 or CE (settled Q5: semantic-leg embedding only)

## Verified facts (grounded)

- **Base pipeline config = `--pipeline`'s**: `rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)` at pipeline.py:245; corpus = `markdown_context_chunks` (PIPELINE_TABLE, pipeline.py:37) with bodies = `coalesce(context,'') || E'\n' || content` (BODY_EXPR, pipeline.py:41); k=K=5 (retrievers.py:13); RRF legs fuse semantic + BM25 at CANDIDATES=50 depth (retrievers.py:15) with RRF_K=60 (retrievers.py:14); CE rerank fed RRF top-RRF_CE_CANDIDATES=10 (retrievers.py:16).
- DB live (psql, re-verified last turn): `markdown_context_chunks` = 250 rows, book `Google Cloud Run`, chunk_index 0-249, context 250/250, embedding 250/250 — `generate_contexts`/`backfill_embeddings` are fast no-ops.
- 10 questions = golden ids 1-10, sequential in file (verified: `[1..10]`), all verified in-subset last turn (ids 1-11 have snippet matches; matches pipeline.md "11/50").
- Reusable internals (import precedent: pipeline.py:26 imports `_get_model`; ragas.py:10 imports `_bm25_corpus`): `_rrf_ranked` retrievers.py:147 (question → fused [(id, score)], incl. empty-run guards), `_bm25_corpus` retrievers.py:76 (triggers + returns shared `(ids, bodies)` cache entry), `_semantic_ranked` retrievers.py:116 (question → [(id, sim)] via `model.encode(question)` + `<=>`), `_cross_encoder_rerank` retrievers.py:181, `_get_cross_encoder` retrievers.py:34, `_get_model` retrievers.py:24.
- `evaluate(retriever, golden, relevant, k=K)` retriever-agnostic harness eval.py:48; `relevant_ids(rows, snippet)` eval.py:38; `load_golden` eval.py:30; eval.run() loops 3 TABLES × 5 retrievers then `EVALS_PATH.write_text(out)` eval.py:109 — this OVERWRITES evals.md wholesale → un-preserved variants section dies on eval re-run (D8 fix).
- evals.md current: `# Eval results` + 15 rows + `_Skipped ..._` line (15 rows are 50-question averages — NOT comparable to 10-question variant rows; non-comparability note in section).
- LLM primitive: `ragas._chat(client, prompt) -> str` ragas.py:45 — qwen3.5:4b (CONTEXT_MODEL pipeline.py:32), temp 0, `think=False` (qwen3.5:4b is a hybrid reasoner — MUST stay `think=False`), num_predict 512 / num_ctx 8192 caps, one retry on `(ollama.ResponseError, httpx.HTTPError)`. Client: `ollama.Client(host=OLLAMA_HOST)` (pipeline.py:163, ragas). Connectivity gate `_connectivity()` pipeline.py:77 (clean exit 1 + stderr message if ollama down / model not pulled).
- main.py current: `ragas` command at main.py:33-41 (implemented since ragas plan); new command slots in before `if __name__` main.py:58.

## Decisions (settled grill + justification)

- D1 (Q1): 10 questions = golden ids 1-10 verbatim. All Cloud Run Runtime chapter, all in-subset. Zero new content.
- D2 (Q2): The 3 new retrievers run over `markdown_context_chunks` ONLY with `BODY_EXPR` — this IS "the base pipeline matching the rrf-ce retriever configuration" (the `--pipeline` table). → 3 new + 1 baseline row, not 9.
- D3 (Q3): Multi-query merge = each leg (original question + up to 3 LLM variants) → `_rrf_ranked` top-`RRF_CE_CANDIDATES`=10 → union candidate (id, body) set → `_cross_encoder_rerank` anchored on ORIGINAL question → top-5. Legs include the ORIGINAL question: degenerate LLM output (empty/duplicate parse) degrades to exactly base behavior.
- D4 (Q4): 4-row `## Query variants` section APPENDED to evals.md, separate from the 15-row section, with explicit non-comparability note (10-question vs 50-question averages).
- D5 (Q5): HyDE: hypothetical doc embedding replaces the query embedding in `_semantic_ranked` leg ONLY; BM25 leg + CE rerank keep original question.
- D6 (Q6): temp 0, recompute each run, no cache file. ~30 short calls ≈ 1-3 min.
- D7 (**not grilled — ASSUMED**): query rewriting feeds the rewritten query to the ENTIRE base `rrf_ce` retriever (BM25 + semantic + RRF + CE rerank all see the rewritten text). Literal definition: "rewrite the query BEFORE it goes to retriever". Multi-query and HyDE anchor CE on the original because the user set that explicitly; rewriting has no such scoping language. Swappable in one line (`_cross_encoder_rerank` gets original q instead) — see Open.
- D8: `eval.py` gains ONLY a section-preserve append (~5 lines + import): without it `eval` re-run (eval.py:109 write) clobbers the variants section — regression of D4. The 15-row metrics/rows stay byte-identical; nothing else in `eval.run()` changes.
- D9: reuse `ragas._chat` as the LLM call helper — no 3rd copy of the chat/retry block (pipeline.py:124-146 and ragas.py:45 would otherwise duplicate).
- D10: new flat module `variants.py` (matches repo layout: eval/pipeline/scorers/ragas) — keeps `retrievers.py` LLM-free.

## Create

- `variants.py` — NEW: prompts, parsers, 3 retriever factories, `run()` harness. ~150 lines.
- `eval.py` — UPDATE: preserve-append variants section (D8). ~6 lines.
- `main.py` — UPDATE: `variants` command + imports. ~8 lines.

## Patterns (existing repo)

Base retriever wiring (pipeline.py:245):

```python
retriever = rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)
```

RRF fusion block to copy for HyDE (retrievers.py:147-169 body of `_rrf_ranked`; empty-run guards + ranx fuse + `(-score, int(id))` sort):

```python
run_sem = Run.from_dict({"q": {str(id): sim for id, sim in sem_list}})
run_bm = Run.from_dict({"q": {str(id): score for id, score in bm_list}})
fused = fuse(runs=[run_sem, run_bm], norm=None, method="rrf", params={"k": RRF_K})
```

CE candidates + body lookup (retrievers.py:203-208 body of `rrf_ce_retriever`):

```python
ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
body_of = dict(zip(ids, bodies))
cands = [(id, body_of[id]) for id, _ in fused]
return _cross_encoder_rerank(ce, question, cands, k)
```

Harness shape (eval.py:72-101 `run()`): golden load → per-question `relevant_ids` → retriever dict → `evaluate` → markdown table lines → write.

LLM call (ragas.py:45 `_chat`, via import):

```python
from ragas import _chat
text = _chat(client, REWRITE_PROMPT.format(question=question))
```

## Exact prompts (module constants in variants.py)

```python
REWRITE_PROMPT = (
    "Rewrite the following question to improve document retrieval quality. Keep "
    "the same meaning but make it more specific and self-contained, preserving key "
    "technical terms. Output only the rewritten question and nothing else.\n"
    "question: {question}"
)

MULTIQUERY_PROMPT = (
    "Generate 3 alternative versions of the following question to improve document "
    "retrieval coverage. Each version must use different wording and synonyms but "
    "preserve the original meaning. Output exactly 3 lines, one question per line, "
    "no numbering and no extra text.\n"
    "question: {question}"
)

HYDE_PROMPT = (
    "Write a short passage (3-5 sentences) that directly answers the following "
    "question, written as it would appear in a technical book about Google Cloud Run. "
    "Output only the passage.\n"
    "question: {question}"
)

N_VARIANTS = 3  # multi-query legs generated per question (plus the original)
```

## Parsing + fallbacks

```python
def _clean_question(line: str) -> str:
    line = line.strip().lstrip("-*• ").strip()
    return re.sub(r"^(?:\d+[.:)])?\s*", "", line).strip()

def parse_multiquery(out: str) -> list[str]:          # public for unit testing
    qs, seen = [], set()
    for line in out.splitlines():
        q = _clean_question(line)
        if q and q.lower() not in seen:
            seen.add(q.lower()); qs.append(q)
    return qs[:N_VARIANTS]

def parse_rewrite(out: str, question: str) -> str:    # empty -> original (D7 leg must run)
    return out.strip() or question

def parse_hyde(out: str, question: str) -> str:      # empty -> embed original question
    return out.strip() or question
```

- rewrite: whole stripped text becomes the query (BM25 `_tokenize` ignores layout; embedder handles newlines).
- multiquery: line split → strip bullets/numbering → dedupe case-insensitive → cap 3. Empty result → legs = [original] only → base behavior.
- hyde: passage text; empty → original question (guard: `model.encode("")` is near-zero vector, garbage ranking).

## Tasks

### CREATE `variants.py`
- DO: module docstring ("Query-transform retrievers: rewrite / multi-query / HyDE over the rrf_ce base (markdown_context_chunks + RRF + cross-encoder), evaluated on golden ids 1-10; LLM = local qwen3.5:4b via ragas._chat").
- DO: imports — `re`; `from eval import load_golden, evaluate, relevant_ids, EVALS_PATH`; `from pipeline import OLLAMA_HOST, BODY_EXPR, PIPELINE_TABLE, _connectivity, ensure_table, generate_contexts, backfill_embeddings`; `from ragas import _chat`; `from retrievers import (RRF_CE_CANDIDATES, RRF_K, _bm25_corpus, _cross_encoder_rerank, _get_cross_encoder, _get_model, _rrf_ranked, _semantic_ranked, _bm25_ranked, rrf_ce_retriever)`; `from ranx import Run, fuse`; `import ollama`; `from setup import connect`.
- DO: constants + prompts + parsers (above).
- DO: retriever factories — all `(conn, table, cache, client, body_expr="content") -> Callable[[str], list[int]]` (drop-in for `evaluate`, eval.py:48):
```python
def rewrite_rrf_ce_retriever(conn, table, cache, client, body_expr="content"):
    base = rrf_ce_retriever(conn, table, cache, body_expr)      # D7: whole base sees rewritten
    def retrieve(question: str, k: int = K) -> list[int]:
        return base(parse_rewrite(_chat(client, REWRITE_PROMPT.format(question=question)), question))
    return retrieve

def multiquery_rrf_ce_retriever(conn, table, cache, client, body_expr="content"):
    ranked = _rrf_ranked(conn, table, cache, body_expr)          # RRF legs (retrievers.py:147)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)    # shared cache entry
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()
    def retrieve(question: str, k: int = K) -> list[int]:
        legs = [question] + parse_multiquery(_chat(client, MULTIQUERY_PROMPT.format(question=question)))
        union: dict[int, str] = {}                                # id -> body, first-seen
        for leg in legs:
            for id, _ in ranked(leg, RRF_CE_CANDIDATES):         # top-10 fused per leg (D3)
                union.setdefault(id, body_of[id])
        if not union:
            return []
        return _cross_encoder_rerank(ce, question, list(union.items()), k)  # ORIGINAL question
    return retrieve

def hyde_rrf_ce_retriever(conn, table, cache, client, body_expr="content"):
    sem = _semantic_ranked(conn, table, _get_model())             # embeds its ARG (retrievers.py:116)
    bm = _bm25_ranked(conn, table, cache, body_expr)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()
    def retrieve(question: str, k: int = K) -> list[int]:
        doc = parse_hyde(_chat(client, HYDE_PROMPT.format(question=question)), question)
        sem_list = sem(doc, CANDIDATES)                           # D5: HyDE embedding leg
        bm_list = bm(question, CANDIDATES)                        # original question
        # fusion block copied from _rrf_ranked body (retrievers.py:158-169), guards included
        if not sem_list:  fused = [(id, s) for id, s in bm_list[:RRF_CE_CANDIDATES]]
        elif not bm_list: fused = [(id, s) for id, s in sem_list[:RRF_CE_CANDIDATES]]
        else:
            run_sem = Run.from_dict({"q": {str(id): sim for id, sim in sem_list}})
            run_bm = Run.from_dict({"q": {str(id): s for id, s in bm_list}})
            fused = fuse(runs=[run_sem, run_bm], norm=None, method="rrf", params={"k": RRF_K})
            fused = sorted(fused.to_dict()["q"].items(), key=lambda kv: (-kv[1], int(kv[0])))
            fused = [(int(d), float(s)) for d, s in fused[:RRF_CE_CANDIDATES]]
        cands = [(id, body_of[id]) for id, _ in fused]
        return _cross_encoder_rerank(ce, question, cands, k)      # ORIGINAL question
    return retrieve
```
  (import `K, CANDIDATES` from retrievers.py:13,15)
- DO: `run()` harness:
```python
VARIANTS_HEADER = "\n## Query variants"   # eval.py imports this marker (D8)

def load_variants_section() -> str:       # eval.py calls this on ITS rewrite
    if not EVALS_PATH.exists(): return ""
    text = EVALS_PATH.read_text()
    return text[text.index(VARIANTS_HEADER):] if VARIANTS_HEADER in text else ""

def run() -> None:
    _connectivity()                                            # pipeline.py:77 gate
    client = ollama.Client(host=OLLAMA_HOST)
    golden = [q for q in load_golden() if q["id"] <= 10]       # D1: ids 1-10
    cache: dict = {}
    with connect() as conn:                                     # setup.py:52
        ensure_table(conn); generate_contexts(conn); backfill_embeddings(conn)   # idempotent, cached
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, content FROM {PIPELINE_TABLE}")             # content ONLY (pipeline.py:242)
            rows = cur.fetchall()
        relevant = [relevant_ids(rows, q["ground_truth_snippet"]) for q in golden]
        mk = lambda name, fn: (name, evaluate(fn, golden, relevant))             # k=K=5
        results = [
            mk("rrf_ce (baseline)", rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)),
            mk("rewrite", rewrite_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
            mk("multiquery", multiquery_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
            mk("hyde", hyde_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
        ]
    lines = [
        f"{VARIANTS_HEADER.lstrip()}: rrf_ce base (markdown_context + BM25/semantic RRF + cross-encoder), "
        f"k=5, golden questions 1-10 (Cloud Run Runtime)",
        "",
        "| Retriever | recall@5 | MRR | MAP |",
        "|---|---|---|---|",
        *[f"| {n} | {m['recall']:.2f} | {m['mrr']:.2f} | {m['map']:.2f} |" for n, m in results],
        "",
        "_NOT comparable to the 50-question averages above — 10-question subset, contextual table only._",
    ]
    prefix = EVALS_PATH.read_text().split(VARIANTS_HEADER)[0].rstrip("\n") if EVALS_PATH.exists() else ""
    EVALS_PATH.write_text(prefix + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"Updated {EVALS_PATH}")
```
  NOTE: implement the `mk` lambda as a plain list comprehension building `(name, metrics)` tuples — lambda shown for brevity only. `evaluate` skips questions with empty `relevant` (eval.py:56-57) — all 10 verified in-subset, expect m['n'] == 10.
- PATTERN: retriever wiring pipeline.py:245; rows select pipeline.py:242; harness eval.py:72-99; `evaluate` eval.py:48; client pipeline.py:163.
- GOTCHA: qwen3.5:4b MUST get `think=False` — inherited from `ragas._chat` (ragas.py:45). Do NOT bypass `_chat` with a raw `client.chat`.
- GOTCHA: `cache` is keyed by table ONLY (retrievers.py:76, one body_expr per table) — all four retrievers here use the same `(PIPELINE_TABLE, BODY_EXPR)` pair, safe; never mix body_exprs on one cache.
- GOTCHA: ranx `Run.from_dict` crashes on empty inner dicts — keep the empty-run guards from retrievers.py:158-162 in the HyDE fusion copy.
- GOTCHA: multiquery/HyDE call `_chat` INSIDE `retrieve` — 10 LLM calls per full eval each, not per leg.
- VALIDATE: `uv run python -c "import variants; print(variants.parse_multiquery('1. Foo\n- Bar\n\nBaz\n4. qux'))"` → `['Foo', 'Bar', 'Baz']`
- MEETS: 3 transform retrievers + baseline, all `question -> top-5 ids`, evaluable by `evaluate`.

### UPDATE `eval.py`
- DO: import `from variants import load_variants_section` (top imports). In `run()`, replace `EVALS_PATH.write_text(out)` (eval.py:109) with:
```python
preserved = load_variants_section()      # "" if absent — D8
if preserved:
    out = out.rstrip("\n") + "\n\n" + preserved
EVALS_PATH.write_text(out)
```
- PATTERN: existing write eval.py:109; section read mirrors ragas-plan's read-modify-write approach.
- GOTCHA: nothing else in `run()` changes — 15-row content string built at eval.py:82-107 is untouched; append happens only when a variants section exists. `eval` with no prior `variants` run produces byte-identical output to today.
- VALIDATE: `uv run python -c "import eval"`; then full regression below.
- MEETS: eval re-runs preserve the variants section; 15 rows unchanged.

### UPDATE `main.py`
- DO: imports: `from variants import run as run_variants`.
- DO: new command between `ragas_command` (main.py:33-41) and `if __name__` (main.py:58):
```python
@app.command(name="variants")
def variants_command() -> None:
    """Query-transform retrieval eval over the pipeline table (markdown_context_chunks,
    golden questions 1-10): rewrite / multi-query / HyDE vs rrf_ce baseline; LLM
    transforms via local qwen3.5:4b (temp 0, recomputed each run); appends the
    ## Query variants section to evals.md."""
    run_variants()
```
- PATTERN: command delegation main.py:19-22 (`eval_command` → `run_eval`).
- VALIDATE: `uv run python main.py --help` → `variants` listed
- MEETS: `python main.py variants` runs the new eval.

## Tests

No pytest — inline `uv run python` heredocs. Seams: parsers are pure (no LLM); retrievers are query-time-LLM (temp 0); `evaluate`/`relevant_ids` reused untouched.

```bash
# T1 parser units (no LLM)
uv run python - <<'PY'
from variants import parse_multiquery, parse_rewrite, parse_hyde
assert parse_multiquery("1. Foo\n- Bar\n\nBaz\n4. qux") == ["Foo", "Bar", "Baz"]
assert parse_multiquery("") == []
assert parse_multiquery("Foo\nfoo\nFOO") == ["Foo"]
assert parse_rewrite("", "orig") == "orig"
assert parse_rewrite("  rewritten q ", "orig") == "rewritten q"
assert parse_hyde("", "orig") == "orig"
print("parser units OK")
PY

# T2 retriever smoke (needs ollama + DB; 3 LLM calls, one question)
uv run python - <<'PY'
import ollama
from pipeline import BODY_EXPR, OLLAMA_HOST, PIPELINE_TABLE, _connectivity
from setup import connect
from variants import rewrite_rrf_ce_retriever, multiquery_rrf_ce_retriever, hyde_rrf_ce_retriever
_connectivity()
client = ollama.Client(host=OLLAMA_HOST)
q = "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
cache = {}
with connect() as conn:
    for name, fn in [
        ("rewrite", rewrite_rrf_ce_retriever),
        ("multiquery", multiquery_rrf_ce_retriever),
        ("hyde", hyde_rrf_ce_retriever),
    ]:
        ids = fn(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)(q)
        assert len(ids) <= 5 and all(isinstance(i, int) for i in ids), (name, ids)
        print(name, "->", ids)
PY
```

T1 expectations: all asserts pass. T2: 3 id lists, each ≤ 5 ints; multiquery ≈ base-or-better on this easy question (semantic leg per variant); hyde ids plausible (not garbage — compare vs `rrf_ce_retriever` on same question).

## Validation (real commands)

- L1 imports: `uv run python -c "import variants, eval, main, retrievers, pipeline, ragas"`
- L2 CLI: `uv run python main.py --help` → `setup / eval / pipeline / ragas / variants`
- L3 units: T1 heredoc above
- L4 smoke: T2 heredoc above
- L5 full: `uv run python main.py variants` → evals.md gains `## Query variants` + 4 rows; stdout prints them; ~30 LLM calls ≈ 1-3 min; rerun → same rows (temp 0, deterministic local embedders/CE)
- L6 regression (critical, D8):
  ```bash
  cp evals.md /tmp/evals.before.md
  uv run python main.py variants                    # section appended
  uv run python main.py eval                        # 15 rows rebuilt + section PRESERVED
  python3 -c "
  a = open('/tmp/evals.before.md').read(); b = open('evals.md').read()
  import sys
  assert b.startswith(a.rstrip('\n')) or open('evals.md').read().split('## Query variants')[0].strip() == a.strip(), 'prefix changed'
  assert '## Query variants' in open('evals.md').read()
  print('evals prefix byte-identical + section preserved')"
  ```
  and with NO variants section present (fresh checkout state): `git stash`-style check — delete `## Query variants` tail from a copy, run `eval`, output must equal pre-change `eval` output byte-for-byte.
- L7 manual: (a) `pkill ollama` then `python main.py variants` → exit 1 with `_connectivity` friendly message; (b) `python main.py ragas` unaffected; (c) `python main.py pipeline` unaffected (pipeline.md unchanged).

## Acceptance

- [ ] `uv run python main.py variants` appends `## Query variants` (4 rows: rrf_ce baseline, rewrite, multiquery, hyde) to evals.md and prints them
- [ ] All four rows computed with base = `rrf_ce` config (markdown_context_chunks + BODY_EXPR + BM25/semantic RRF + CE rerank, k=5) on golden ids 1-10
- [ ] recall@5 / MRR / MAP per row (three metrics each), via untouched `evaluate` (eval.py:48)
- [ ] Multi-query: legs = original + ≤3 variants, per-leg RRF top-10 → union → CE on ORIGINAL question (D3)
- [ ] HyDE: hypothetical doc embedding in semantic leg ONLY; BM25 + CE keep original question (D5)
- [ ] Rewrite: rewritten query feeds the entire base retriever (D7, ASSUMED)
- [ ] `python main.py eval` 15-row section byte-identical; variants section preserved across eval re-run (L6)
- [ ] No new deps (uv.lock untouched); all LLM via ollama qwen3.5:4b `think=False` temp 0
- [ ] L1-L7 green

## Open / assumed

- ASSUMED — D7: query-rewrite CE rerank sees the REWRITTEN query (whole-retriever reading). Multi-query/HyDE anchor CE on the ORIGINAL (explicit user decisions). If rewrite should also anchor CE on the original, one-line change in `rewrite_rrf_ce_retriever` (build custom closure instead of delegating to base) — flag before implementing.
- ASSUMED — A2 (inherited): ollama running + `qwen3.5:4b` pulled; `markdown_context_chunks` populated (ensured idempotently inside `run()`).
- ASSUMED — A3: HyDE docs + rewritten queries fit `NUM_CTX=8192` (ragas._chat shared caps) — transform outputs are ≤ a few hundred tokens.
- Open: multi-query variant count N=3 (MULTIQUERY_PROMPT says exactly 3) — prompt constant; changing it = prompt + N_VARIANTS.
- Open: if `qwen3.5:4b` wraps answers in preamble despite "Output only ...", parsers still hold (strip/line-split takes first lines; rewrite/hyde use whole text — tolerant). If observed degradation, tighten prompts, not parsers.

## Notes

- Rejected: RAGAS library / LangChain MultiQueryRetriever — new deps, violates constraint; hand-impl ≈ 150 lines total.
- Rejected: editing `retrievers.py` to add LLM legs — pollutes the LLM-free deterministic module; variants.py copies the 12-line fusion block instead (cite retrievers.py:158-169) once.
- Rejected: HyDE doc fed to BM25 as well — user decision Q5 scopes to embedding leg only.
- Rejected: caching transforms to disk (settled Q6 — recompute; 30 short calls).
- Rejected: separate variants.md output file — user decision Q4: section in evals.md; D8 preserve-append is the only way both "in evals.md" and "eval re-run must not regress" hold.
- Tradeoff: `eval.py` import of `variants` pulls ollama into eval's import chain — already installed transitively (pipeline ← main), no new failure mode; alternative (marker-string duplicated in eval.py) risks drift.
- Tradeoff: union-of-legs multi-query loses per-leg RRF scores in the union (CE re-scores from scratch on original question) — by design (D3); keeps CE the single final ranker, matching base config.
- Sequencing risk: none — 3 files, one new module; `retrievers.py`/`scorers.py`/`golden_set.JSON` untouched. Runtime risk: ollama down (gated), 4b prompt compliance (fallbacks in every parser).

## Amendments
(append-only: date - change - why)

- 2026-09-18 - `VARIANTS_HEADER = "## Query variants"` WITHOUT leading `\n` (plan had `"\n## Query variants"`) — simplifies prefix split/section re-join, avoids double blank line when eval.py re-appends `out + "\n\n" + preserved`.
- 2026-09-18 - eval.py imports `load_variants_section` LAZILY inside `run()` (plan had top-level import) — top-level would be circular: variants.py imports `evaluate`/`load_golden` from eval.py, eval.py importing variants at top deadlocks (`ImportError` on partially initialized eval). Lazy at call site: eval fully loads first in every entry order (`main.py` imports eval before variants; `import variants` alone works since eval completes before its run() body runs).
- 2026-09-18 - `rewrite` baseline scope D7 confirmed with user before implementation ("implement" = standard reading: rewritten query feeds whole base retriever).
