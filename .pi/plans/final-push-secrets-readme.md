# Plan: final push prep — secrets audit, .gitignore hardening, README rewrite

Implements: user request (final-push prep + README Write Up / Documentation) | Type: docs + repo hygiene | Complexity: L | Systems: git, README.md | Deps: none

## Goal
As project owner, I want the repo safe to publish (no secrets, .env + langfuse/ ignored) with a README containing a Write Up (evals, models, methods, results, tracing) and Documentation (exploratory blurb, tech overview, Typer commands, architecture diagram), so the final push is clean and presentable.

## Secrets audit — RESULTS (already done, no remediation of history needed)

- `git log --oneline --all` → **zero commits**. `git ls-files` → empty. Nothing tracked, nothing in history. → No untracking, no history rewrite, no forced rotation strictly required.
- `.env` (172 bytes) contains exactly: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL` (keys for a **localhost** Langfuse stack — low blast radius, but still must never be committed).
- Regex scan (`sk-…`, `api_key=`, `secret=`, `Bearer …`, `ghp_…`, `AKIA…`) over all project `.py/.md/.toml/.json` (excluding `langfuse/`, `.venv/`): **zero hits**.
- `setup.py:25` — `DB_URL = "postgres://postgres:password@localhost/p2"` — hardcoded local default creds. Not a real secret (local Postgres default); out of scope to change, note in README Documentation only if desired.
- `langfuse/` = full upstream Langfuse repo clone, **has its own `.git/` dir**, contains only `.env*.example` files (no real `.env`). Its `docker-compose.yml` was locally modified (Nov 12 11:56). Already gitignored — stays untracked.
- `trace_images/` — 3 PNGs of Langfuse trace UI (`p2_example_rag_ask_stream_trace.png`, `p2_rag_ask_rerank.png`, `p2_rag_ask_retrieve.png`). Visual inspection needed (Task 3) for incidental secrets in screenshots; otherwise intended to be committed and referenced by README.
- `.pi/` — plans + langfuse skill files; skill reads credentials from env vars only, no secrets found. Intended to be committed.

## Out of scope
- Not: changing any `.py` source, `evals.md`, `pipeline.md`, `golden_set.JSON`
- Not: rotating Langfuse keys (localhost instance; harmless once never committed)
- Not: removing the `langfuse/` clone from disk (gitignored is enough)
- Not: pushing / creating the actual commit (plan only; ask user before executing any of this)

## Read first (required)
- `evals.md` (whole file, 34 lines) — the exact score tables the README Write Up must reproduce
- `pipeline.md` (whole file) — contextual-retrieval pipeline score (11/50-question subset caveat)
- `main.py:27-71` — all 7 Typer commands + their docstrings (README Documentation)
- `main.py:88-181` — SSE streaming + Langfuse observation tree (README tracing/streaming)
- `traces.py:1-40` — `@observe` decorator pattern, `load_dotenv`, Langfuse client init
- `pipeline.py:25-58` — contextual retrieval constants + Anthropic `PROMPT_TEMPLATE`
- `ragas.py:1-38` — faithfulness prompts, `QUESTIONS`, statement extract/verify
- `variants.py:1-10` — rewrite / multiquery / HyDE design notes (D3/D5/D7)
- `scorers.py` (whole file) — recall@k, MRR, AP definitions
- `README.md` (current, 78 lines) — existing content to restructure, not discard
- `.gitignore` (current, 14 lines)

## Create / Modify
- `.gitignore` — UPDATE: broaden env-file ignoring (Task 1)
- `README.md` — REWRITE into two sections per user spec (Task 4)
- `.pi/plans/final-push-secrets-readme.md` — this plan

## Patterns
Current `.gitignore` (verbatim):
```
# Secrets + upstream reference clone
.env
langfuse/
```

## Tasks

### Task 0 — Decision gate (ASK USER BEFORE EXECUTING)
User global instruction: never implement a plan without asking. Confirm:
1. Proceed with Tasks 1-4?
2. "MMP" in your request: repo computes **MRR** (scorers.py:14) and **MAP** (scorers.py:23). ASSUMED — MMP = MRR.
3. Root `PLAN.md` = stale plan for the already-completed chunker-table restructure. Delete it, or leave? Recommend: delete.
4. `.pi/` (plans + skills) — commit or ignore? Recommend: commit (no secrets).
5. Commit `trace_images/` and reference PNGs from README tracing section? Recommend: yes.

### Task 1 — UPDATE `.gitignore` (hardening)
- DO: replace `.env` line with:
```
# Secrets + upstream reference clone
.env
.env.*
!.env.example
langfuse/
```
  (`.env.*` catches `.env.local`/`.env.production` etc.; `!.env.example` allows a future sanitized example. `langfuse/` stays — trailing-slash dir pattern, matches clone incl. its nested `.git`.)
- Also append `.python-version`? NO — pyenv pin file, fine to commit.
- VALIDATE: `git check-ignore -v .env && git status --porcelain | grep -c 'langfuse'` (expect first prints the rule, second `0`)
- MEETS: no `.env*` variant or `langfuse/` path can be staged by `git add .`

### Task 2 — Pre-commit secrets re-scan
- DO: re-run over the full working tree that a `git add .` would stage:
```bash
git add -A --dry-run > /tmp/staged.txt
grep -E '\.env($|\.)|langfuse/' /tmp/staged.txt && echo FAIL || echo OK
grep -rniE 'sk-[a-zA-Z0-9]{20,}|api[_-]?key\s*=\s*["'"'"'][^"'"'"']{8,}|secret\s*=\s*["'"'"'][^"'"'"']{8,}|ghp_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{12,}' \
  --include='*.py' --include='*.md' --include='*.toml' --include='*.json' --include='*.JSON' \
  --exclude-dir=langfuse --exclude-dir=.venv --exclude-dir=.git . && echo FAIL || echo OK
```
- MEETS: both print `OK` (expect: audit above says they will)
- NOTE: `git add -A --dry-run` output is informational; do NOT actually stage in this task.

### Task 3 — Visual check `trace_images/` (screenshots)
- DO: open each PNG (`trace_images/p2_example_rag_ask_stream_trace.png`, `p2_rag_ask_retrieve.png`, `p2_rag_ask_rerank.png`), confirm no API keys visible in Langfuse UI screenshots (keys are masked in Langfuse UI; expected clean).
- MEETS: no secret visible; PNGs approved for commit + README embedding.

### Task 4 — REWRITE `README.md` (the bulk of the work)
- DO: full rewrite, two top-level sections exactly as user specified. Keep existing prerequisites block (README.md:9-19) and command list (fold into Documentation). Structure with concrete content:

```
# p2 — RAG retrieval pipeline
(one-paragraph intro, keep current README.md:3-8 gist)

## Write Up
### Chunkers & retrieval eval
- 3 chunkers (chunkers.py): fixed (800-char windows, 100 overlap), recursive
  (nested separators "\n\n"/"\n"/". "/" ", greedy merge ≤800), markdown
  (ATX-heading section-aware, paragraph merge ≤800, heading-path metadata +
  char/line provenance via MarkdownChunk)
- 5 retrievers (retrievers.py): ts_rank_cd (Postgres GIN tsvector), BM25
  (rank_bm25), semantic (pgvector HNSW, top-50 candidates), RRF fusion (ranx,
  k=60), RRF + cross-encoder rerank (top-10 RRF → ms-marco cross-encoder → top-5)
- TABLE: copy evals.md 15-row table verbatim (k=5, 50 golden questions,
  snippet-containment relevance)
- Query variants table (rewrite / multiquery / HyDE vs rrf_ce baseline,
  questions 1-10) — copy verbatim + its "NOT comparable" footnote

### Models
- LLM: qwen3.5:4b via local ollama (contexts, answers, query transforms, RAGAS judge)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim → VECTOR(384))
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L12-v2 (BERT, 512 tokens, ~127MB)

### Query parsing / contextual retrieval
- Anthropic recipe (pipeline.py:43-47 PROMPT_TEMPLATE): whole doc + chunk →
  qwen3.5:4b generates short situating context (temp 0, ≤128 tokens, 1 worker,
  batch-32 commit, cached in markdown_context_chunks, never dropped)
- Context prepended for BM25 corpus, embeddings, and CE rerank passages
  (BODY_EXPR D1: coalesce(context,'') || E'\n' || content); original content
  kept for relevance containment
- LLM query transforms (variants.py): rewrite (feeds whole rrf_ce), multi-query
  (union RRF legs, CE anchored on ORIGINAL question), HyDE (hypothetical-doc
  embedding replaces query embedding in semantic leg only)

### Metrics
- scorers.py: recall@5, MRR, MAP (definitions in file) — mean over non-skipped
  golden questions; relevance = ground-truth-snippet containment, 60-char prefix
  fallback (eval.py:34-41)
- RAGAS-style faithfulness (ragas.py): 5 hard-coded questions → grounded answer →
  statement extraction → verify each against context → F = supported/total,
  reference-free, qwen3.5:4b as judge
- Pipeline result (pipeline.md): rrf_ce + markdown_context → recall@5 / MRR /
  MAP all 1.00 — CAVEAT: 11/50-question Cloud Run subset, 39 skipped; not
  comparable to the 50-question table

### Best pipeline setup and why
- Best full-corpus: rrf_ce + recursive chunker (recall 0.94, MRR/MAP 0.81);
  markdown close second (0.92 / 0.81). Contextual retrieval (markdown_context)
  hits 1.00 on its subset.
- Why: RRF fixes BM25/semantic rank disagreement; cross-encoder is the single
  biggest MRR jump (0.66→0.83 fixed, 0.81 across chunkers) because it scores
  query-passage jointly instead of lexical/embedding proxy; query transforms
  (rewrite/multiquery/HyDE) added nothing over the already-strong baseline
  (1.00 baseline vs 0.93-1.00 variants on subset)

### Tracing with LangFuse + streaming
- traces.py: @observe decorator chain rag-ask (span) → retrieve (retriever) →
  rerank (span) → generate (generation) with usage_details from ollama token
  counts; langfuse SDK get_client(), .env via python-dotenv, localhost:3000 stack
- main.py serve: FastAPI POST /ask → SSE stream (status → token deltas → done);
  observations built manually (start_observation) around threadpool
  retrieve/rerank; TTFT via completion_start_time on first token; disconnect +
  error paths recorded (WARNING/ERROR levels) then re-raise
- Embed the 3 trace_images/ PNGs here

## Documentation
### (blurb) Exploratory project
- Not practical/production: local-only stack (Postgres creds default, ollama
  on localhost, Langfuse self-hosted), tiny two-book corpus, snippet-containment
  relevance proxy, hard-coded RAGAS questions, exploratory eval harness —
  built to learn/compare RAG techniques, not to deploy

### Technologies used
- Python 3.12 + uv; typer CLI; fastapi + uvicorn (SSE); psycopg3 + pgvector
  (HNSW) + tsvector/GIN; rank-bm25; ranx (RRF fusion); sentence-transformers
  (embeddings + cross-encoder); ollama (qwen3.5:4b); langfuse SDK (tracing);
  numpy; python-dotenv

### Typer commands
- Keep/refine current README command section: `uv run main.py setup | eval |
  pipeline | ragas | variants | trace "<question>" | serve` with one-line purpose
  each (from main.py docstrings) + prerequisites block from current README

### Architecture diagram
- ASCII/mermaid diagram, flow:
  books (letsgo_further.md, cloud_run.md) → chunkers (fixed/recursive/markdown)
  → markdown_context_chunks table (+ qwen3.5:4b contextual contexts, embeddings
  VECTOR(384), generated tsvector) → retrieval: BM25 ∥ semantic → RRF (ranx,
  k=60, 50 candidates) → cross-encoder rerank (top-10→5) → qwen3.5:4b cited
  answer → SSE stream (FastAPI) / Langfuse trace
- Also note eval side-path: golden_set.JSON → evaluate → evals.md
```
- PATTERN: score tables copied verbatim from `evals.md:3-17` and `evals.md:21-29`; command docstrings from `main.py:27-71`; keep `_Skipped_` footnotes
- GOTCHA: never round/re-derive scores — copy verbatim; keep the "NOT comparable" caveat (10-question subset) or the Write Up misleads
- GOTCHA: mermaid renders on GitHub; if user prefers plain image, embed ASCII instead — ASK (Task 0)
- MEETS: both sections contain every bullet from the user's request; every number traceable to `evals.md`/`pipeline.md`; commands match `main.py` exactly

### Task 5 — Final validation
- DO:
```bash
git check-ignore -v .env langfuse/          # both matched by ignore rules
git status --porcelain | grep -E '^ .*(\.env|langfuse)' && echo FAIL || echo OK
npx markdown-link-check README.md 2>/dev/null || true   # or manual click-through
```
- Manual: render README.md (GitHub preview) — tables aligned, images resolve (`trace_images/...` relative paths), mermaid renders
- MEETS: repo safe to `git add . && git commit` with zero secret exposure; README complete per spec

## Tests
Docs-only change — no unit tests. Validation = ignore-rule checks (Task 1/5), secret scans (Task 2), visual screenshot check (Task 3), README render check (Task 5).

## Validation (real cmds)
L1 `git check-ignore -v .env langfuse/` · L2 secret grep scan (Task 2 cmd) · L3 `git add -A --dry-run` staged-list audit · L4 manual: GitHub-style markdown render of README.md

## Acceptance
- [ ] `.env`, any `.env.*`, and `langfuse/` provably ignored (`git check-ignore`)
- [ ] `git add -A --dry-run` staged list contains no env/langfuse/secret hits
- [ ] README has exactly two main sections: "Write Up" and "Documentation"
- [ ] Write Up covers: chunker/retriever score table, models, contextual retrieval + query parsing, recall@5/MRR/MAP + RAGAS faithfulness, best-pipeline rationale, LangFuse tracing + streaming
- [ ] Documentation covers: exploratory blurb, tech overview, all 7 Typer commands, architecture diagram
- [ ] All scores verbatim from evals.md / pipeline.md, caveats preserved
- [ ] Zero source-code files modified

## Open / assumed
- ASSUMED — "MMP" = MRR (repo has recall@5, MRR, MAP; no metric named MMP)
- ASSUMED — root PLAN.md (stale restructure plan) gets deleted or left as-is pending Task 0 answer
- ASSUMED — `.pi/` and `trace_images/` are committed (no secrets found in audit)
- Open: delete stale root PLAN.md? (recommend yes)
- Open: `.pi/` committed or ignored? (recommend committed)
- Open: mermaid vs ASCII architecture diagram? (recommend mermaid — GitHub renders it)
- Open: keys in .env point at localhost Langfuse — rotate anyway before any public push? (recommend: unnecessary if never committed, cheap to rotate if paranoid)

## Notes
- Repo has zero commits — the entire "remove exposed secrets" ask reduces to gitignore hygiene BEFORE the first commit; no `git filter-repo`/BFG/rotation needed. If the repo were pushed before this plan ran, escalation would be: untrack (`git rm --cached`) + history rewrite + rotation.
- Rejected: adding `.env.example` skeleton in this pass — user didn't ask; README already documents required vars (README.md:16-19).
- Rejected: touching `setup.py` DB_URL — local default creds, out of surgical scope.
- Sequencing risk: Task 4 is large but single-file; Tasks 1-3 independent, can go first in any order.
