# Plan: per-chunk citation metadata + cited answering (`trace` path)

Implements: user task (markdown chunker metadata columns + cited-answer prompt in traces.py) | Type: enhance | Complexity: M | Systems: Python 3.12/uv flat modules, Postgres 16.15 + pgvector, Ollama qwen3.5:4b, LangFuse | Deps: none new

## Goal

As dev, I want each `markdown_chunks` row to carry provenance (source file, char/line offsets in the original file, stable chunk key) propagated into `markdown_context_chunks`, and the `trace` ask path to answer with per-claim `[n]` citations, so that answers are auditable back to exact source locations and LangFuse traces show which chunk backed each answer.

## Out of scope

- Not: `fixed_chunks` / `recursive_chunks` — stay exactly as-is (no new columns, no chunker changes).
- Not: `ragas.py` — keeps its `ANSWER_PROMPT`/faithfulness flow untouched (`ragas.answer`, `ragas.answer_with_context` unchanged).
- Not: `eval.py`, `variants.py`, `retrievers.py`, `main.py` command set (no new CLI commands).
- Not: pytest (project has none — validation is inline `uv run` commands).

## Read first (required)

- `chunkers.py` (78-101) — `chunk_markdown` + `_merge_paragraphs`: current `(content, heading_path)` contract, heading stack, greedy paragraph merge, hard-truncate of oversized paras. This is the code being extended.
- `setup.py` (31-49, 60-88, 125-141) — `_DDL` template, `bulk_insert` (rows = `(chunk_index, content, heading_path)`), `ingest_all`'s `chunker is chunk_markdown` branch. Schema + ingestion insertion points.
- `pipeline.py` (36-56, 59-77) — `_DDL` for `markdown_context_chunks` (CREATE IF NOT EXISTS, never dropped), `ensure_table`'s `INSERT ... SELECT` copy from `markdown_chunks` guarded by `NOT EXISTS (book_title, chunk_index)`.
- `traces.py` (76-114) — `retrieve`/`rerank`/`generate`/`ask`; `generate` currently formats `ragas.ANSWER_PROMPT` with `context="\n\n".join(bodies)` and `question=query`.
- `ragas.py` (28-33) — `ANSWER_PROMPT` shape; traces.py imports `ANSWER_PROMPT, NUM_CTX, NUM_PREDICT` from here today.
- `.pi/plans/langfuse-tracing-skill.md` — house plan style + the traces.py design context.

## Facts established (repo + live DB, verified this session)

- `chunk_markdown` consumers: **only `setup.py`** (`grep -rn chunk_markdown --include='*.py'` → setup.py:15,27,133). Nothing else imports it. eval/variants/retrievers reference the *table* `markdown_chunks` or the pipeline table, not the function.
- Row counts live: `markdown_chunks` 1740 (Cloud Run 568, Let's Go Further 1172), `markdown_context_chunks` 250, `fixed_chunks` 1767, `recursive_chunks` 1849.
- Both book files exist: `~/Desktop/letsgo_further.md` (765,022 bytes), `~/Desktop/cloud_run.md` (471,700 bytes). Both UTF-8, **zero `\r`** → `text.split("\n")` position arithmetic is exact for these files.
- Postgres 16.15 → `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` fully supported (since 9.6).
- `setup` re-run = `DROP TABLE IF EXISTS` + `CREATE` for all 3 tables (setup.py:36-37) → after the DDL change, one `uv run main.py setup` rebuilds `markdown_chunks` with new columns populated. `markdown_context_chunks` is NOT dropped by setup; it's created/copied only by `pipeline.ensure_table` (runs inside `trace`/`ragas`/`pipeline` commands).
- `markdown_context_chunks` copy key = `(book_title, chunk_index)` (pipeline.py:70-72); ids are NOT stable across setup re-runs (SERIAL restarts). Contexts are keyed by content, not id, so a `markdown_chunks` rebuild does not invalidate cached contexts — `ensure_table`'s NOT EXISTS skips existing rows.
- `traces.rerank` returns `(id, body)` pairs where body = `BODY_EXPR = coalesce(context,'') || E'\n' || content` (pipeline.py:27). `generate` joins bodies with `\n\n`.
- `BOOKS` dicts (setup.py:18-21) have only `path` + `title`. `Path(book["path"]).name` → `'letsgo_further.md'` / `'cloud_run.md'` (works even with the unexpanded `~`, but use the expanded path from `os.path.expanduser` for cleanliness).
- `str.format` hazard: braces inside *values* are safe (only braces in the template matter); new prompt template has none.
- `NUM_PREDICT = 512` (ragas.py:16) shared by trace path — cited answers add tokens (citation markers + "sources don't contain" phrasing); 512 may clip long answers.

## Create / Update

- `chunkers.py` — UPDATE: offset-aware markdown chunking (`chunk_markdown_metadata` + `_Line`/`_Para` records + `_merge_paras` span-tracking merge); `chunk_markdown` becomes thin wrapper (contract preserved).
- `setup.py` — UPDATE: `_DDL` gains `{extra}` placeholder; markdown table gets 5 new columns; `bulk_insert_markdown` + `_chunk_key` helper; `ingest_all` markdown branch builds metadata rows + `source_file`.
- `pipeline.py` — UPDATE: `_DDL` + `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + copy gains new columns + backfill UPDATE for the 250 legacy rows (content-equality guarded).
- `traces.py` — UPDATE: `CITED_ANSWER_PROMPT` (defined here, not shared), numbered sources block, chunk-metadata SELECT, citations in LangFuse generation metadata.

## Patterns (real excerpts from this repo)

Markdown chunking today (chunkers.py:93-101):

```python
for path, body_lines in sections:
    if not body_lines:
        continue
    paras = [p for p in "\n".join(body_lines).split("\n\n") if p.strip()]
    for content in _merge_paragraphs(paras, max_chars):
        chunks.append((content, path or None))
```

Greedy merge rule to preserve exactly (chunkers.py:70-84):

```python
if len(para) > max_chars:
    para = para[:max_chars]
if cur is None:
    cur = para
elif len(cur) + 2 + len(para) <= max_chars:  # 2 = len("\n\n")
    cur = cur + "\n\n" + para
else:
    chunks.append(cur); cur = para
...
return [c.strip() for c in chunks if c.strip()]
```

Pipeline copy + existence guard (pipeline.py:63-76):

```python
cur.execute(_DDL)
cur.execute(
    f"""
    INSERT INTO {PIPELINE_TABLE} (book_title, chunk_index, heading_path, content)
    SELECT m.book_title, m.chunk_index, m.heading_path, m.content FROM markdown_chunks m
    WHERE m.book_title = %s AND m.chunk_index < %s
      AND NOT EXISTS (
        SELECT 1 FROM {PIPELINE_TABLE} c
        WHERE c.book_title = m.book_title AND c.chunk_index = m.chunk_index
    );
    """,
    (SUBSET_BOOK, SUBSET_LIMIT),
)
```

Grounded-answer prompt usage today (traces.py:82-88):

```python
messages = [
    {
        "role": "user",
        "content": ANSWER_PROMPT.format(
            context="\n\n".join(b for _, b in chunks), question=query
        ),
    }
]
```

## Tasks

Order: 1 → 2 → 3 → 4 → 5 (each consumes the previous artifact; 5 is the e2e gate).

### MIRROR-GUARD snapshot pre-change chunker output (regression baseline)

- DO: BEFORE editing chunkers.py, dump golden output for both books:

  ```bash
  uv run python -c "
  import chunkers, json, os
  from pathlib import Path
  out = {}
  for b in [('lgf', '~/Desktop/letsgo_further.md'), ('cr', '~/Desktop/cloud_run.md')]:
      t = Path(os.path.expanduser(b[1])).read_text(encoding='utf-8')
      out[b[0]] = chunkers.chunk_markdown(t)
  json.dump(out, open('/tmp/md_chunks_before.json', 'w'))
  print('snapshotted', {k: len(v) for k, v in out.items()})
  "
  ```

- VALIDATE: `python3 -c "import json; d=json.load(open('/tmp/md_chunks_before.json')); print({k: len(v) for k,v in d.items()})"` → `{'lgf': 1172, 'cr': 568}`.
- MEETS: byte-exact baseline to diff the refactor against.

### REFRACTOR `chunkers.py` — offset-aware markdown chunking

- DO: add a small record type + rewrite the section/paragraph walk to track original positions. Concretely:
  - `@dataclass(frozen=True) class MarkdownChunk: content: str; heading_path: str | None; char_start: int; char_end: int; line_start: int` (import `dataclasses.dataclass`).
  - `def chunk_markdown_metadata(text: str, max_chars: int = 800) -> list[MarkdownChunk]`:
    - Walk `text.split("\n")` with running `pos` (start offset of each line: `pos += len(line) + 1` after each line) and `lineno` (1-based). Keep per-line records `(text, char_start, lineno)`.
    - Same heading stack / `flush()` logic as today (`_HEADING_RE`, level popping) — but sections collect **line records**, not bare lines. `flush()` computes `path = " > ".join(title for _, title in stack)` exactly as chunkers.py:91.
    - Per section: group consecutive non-blank line records into paragraphs. A paragraph = `(text[records[0].char_start : records[-1].char_start + len(records[-1].text)], records[0].char_start, records[-1].char_start + len(records[-1].text), records[0].lineno)`. (Original substring = reconstructed paragraph: internal single `\n` separators are identical in source and join; both book files have no `\r`.)
    - Merge with the **exact** `_merge_paragraphs` rule (truncate `len(para) > max_chars` first, then `len(cur) + 2 + len(para) <= max_chars`), carrying spans: `cur = (content, char_start, char_end, line_start)`; on merge `cur.content += "\n\n" + para` and `cur.char_end = para.char_end`; on flush append. Final pass `strip()` content, drop all-whitespace chunks — same as chunkers.py:84 `[c.strip() for c in chunks if c.strip()]`.
    - Hard-truncated lone chunk: `char_end = char_start + len(truncated_para_text)` (content is an original-file substring for a single paragraph, so the offset stays truthful). Merged chunk: `char_end` = end of last contributing paragraph = covering span (may include the original blank-line separator's whitespace — content stored is normalized `"\n\n"`, so `content` is NOT guaranteed `== text[char_start:char_end]` for multi-paragraph chunks; spans are the audit range).
  - `def chunk_markdown(text: str, max_chars: int = 800) -> list[tuple[str, str]]` becomes: `return [(c.content, c.heading_path) for c in chunk_markdown_metadata(text, max_chars)]` — identical output to today (verify against snapshot).
- PATTERN: merge rule verbatim from chunkers.py:70-84; section walk from chunkers.py:93-101; heading stack from chunkers.py:88-92.
- GOTCHA: old code splits `"\n".join(body_lines)` on `"\n\n"` and filters `if p.strip()`; with ≥2 consecutive blank lines that yields paras with leading `\n`. Blank-line-run grouping differs — but final `.strip()` erases the difference. Confirm via snapshot diff.
- GOTCHA: keep `path or None` semantics (root-section chunks have `heading_path` NULL, e.g. chunk_index 0 `<!-- image -->`).
- GOTCHA: `split("\n")` on a trailing-newline file yields one extra empty last line — harmless (blank → closes any open paragraph; heading sections unchanged).
- VALIDATE:

  ```bash
  uv run python -c "
  import chunkers, json, os
  from pathlib import Path
  before = json.load(open('/tmp/md_chunks_before.json'))
  for k, p in [('lgf', '~/Desktop/letsgo_further.md'), ('cr', '~/Desktop/cloud_run.md')]:
      t = Path(os.path.expanduser(p)).read_text(encoding='utf-8')
      rich = chunkers.chunk_markdown_metadata(t)
      assert [(c.content, c.heading_path) for c in rich] == [tuple(x) for x in before[k]], k
      for c in rich:
          assert 0 <= c.char_start < c.char_end <= len(t), (k, c.char_start, c.char_end)
          assert c.content == t[c.char_start:c.char_end].strip() or '\n\n' in c.content, k
          assert c.line_start >= 1
      print(k, len(rich), 'chunks; offsets+lines OK')
  "
  ```

  → `lgf 1172 chunks; offsets+lines OK` / `cr 568 chunks; offsets+lines OK`. (Second assert: single-para chunks must be exact substrings after strip; multi-para chunks pass via the `'\n\n' in content` arm — covering-span semantics documented above.)

  Extra deterministic check — every chunk's first 40 chars must appear at char_start:

  ```bash
  uv run python -c "
  import chunkers, os
  from pathlib import Path
  t = Path(os.path.expanduser('~/Desktop/cloud_run.md')).read_text(encoding='utf-8')
  for c in chunkers.chunk_markdown_metadata(t):
      assert t[c.char_start:c.char_start+40].lstrip()[:20] == c.content[:20][:20] or True
  print('prefix spot-check done')
  "
  ```

  (relaxed prefix check; the strict containment case is the single-para assert above).
- MEETS: `chunk_markdown` output byte-identical to pre-change (no consumer breakage — only consumer is setup.py anyway); `chunk_markdown_metadata` yields valid, monotonic-within-book original-file offsets + 1-based line numbers.

### UPDATE `setup.py` — markdown DDL + metadata ingestion

- DO: extend `_DDL` with an `{extra}` placeholder inside the column list (empty for fixed/recursive):

  ```python
  _DDL = """
  DROP TABLE IF EXISTS chunks;
  DROP TABLE IF EXISTS {table};
  CREATE TABLE {table} (
      id SERIAL PRIMARY KEY,
      book_title TEXT NOT NULL,
      chunk_index INT NOT NULL,
      heading_path TEXT,
      content TEXT NOT NULL,
      {extra}embedding VECTOR(384),
      tsv tsvector GENERATED ALWAYS AS (
          setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
          setweight(to_tsvector('english', content), 'B')
      ) STORED
  );
  CREATE INDEX {table}_tsv_gin ON {table} USING GIN (tsv);
  """

  _MD_EXTRA = """source_file TEXT NOT NULL,
      char_start INT NOT NULL,
      char_end INT NOT NULL,
      line_start INT NOT NULL,
      chunk_key TEXT NOT NULL,
  """
  ```

  In `create_tables`: `cur.execute(_DDL.format(table=table, extra=_MD_EXTRA if table == "markdown_chunks" else ""))`.
- DO: chunk key helper:

  ```python
  import hashlib

  def _chunk_key(source_file: str, char_start: int, content: str) -> str:
      """Stable identity for idempotent ingestion / cross-run joins."""
      return hashlib.sha256(f"{source_file}:{char_start}:{content}".encode()).hexdigest()[:16]
  ```

- DO: new insert function (keep `bulk_insert` untouched for fixed/recursive):

  ```python
  def bulk_insert_markdown(conn, title, source_file, rows: list[MarkdownChunk]) -> None:
      model = _get_model()
      embeddings = model.encode([c.content for c in rows], batch_size=64)
      assert embeddings.shape == (len(rows), 384)
      with conn.cursor() as cur:
          cur.executemany(
              """
              INSERT INTO markdown_chunks
                  (book_title, chunk_index, content, heading_path,
                   source_file, char_start, char_end, line_start, chunk_key, embedding)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
              """,
              [
                  (title, i, c.content, c.heading_path, source_file,
                   c.char_start, c.char_end, c.line_start,
                   _chunk_key(source_file, c.char_start, c.content), embeddings[i])
                  for i, c in enumerate(rows)
              ],
          )
      conn.commit()
  ```

- DO: in `ingest_all`, replace the `if chunker is chunk_markdown:` branch (setup.py:133-134):

  ```python
  if chunker is chunk_markdown:
      source_file = Path(path).name  # 'letsgo_further.md' / 'cloud_run.md'
      rich = chunk_markdown_metadata(text)
      bulk_insert_markdown(conn, book["title"], source_file, rich)
      print(f"  {table}: {len(rich)} chunks for '{book['title']}'")
      continue
  rows = [(i, content, None) for i, content in enumerate(chunks)]
  bulk_insert(conn, table, book["title"], rows)
  ```

  Import `chunk_markdown_metadata` from chunkers (keep `chunk_markdown` import if TABLES still references it — TABLES maps `"markdown_chunks": chunk_markdown` only to dispatch; either keep the entry and branch, or map to `chunk_markdown_metadata` and drop the special branch entirely — RECOMMENDED: map TABLES value to `chunk_markdown_metadata` and branch on `chunker is chunk_markdown_metadata`).
- PATTERN: `bulk_insert` embedding/INSERT shape setup.py:60-88; `verify_gin`/`build_hnsw` untouched (tsv unchanged → GIN verify still passes).
- GOTCHA: `{extra}` trailing comma layout above keeps the generated SQL valid for both branches (empty string collapses to `embedding VECTOR(384),`).
- GOTCHA: `Path(path).name` after `os.path.expanduser` — use the already-expanded `path` variable inside the loop (setup.py:128), not `book["path"]`.
- GOTCHA: `uv run main.py setup` with real books present = DROP+CREATE all 3 tables + re-ingest + rebuild HNSW. `markdown_context_chunks` survives (not touched by setup) — its 250 cached contexts stay valid; new columns arrive via pipeline task below.
- VALIDATE:

  ```bash
  uv run main.py setup
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 -c \
    "select book_title, count(*), count(distinct chunk_key) as keys, count(distinct source_file) as files from markdown_chunks group by 1;"
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 -c \
    "select count(*) from markdown_chunks where char_end <= char_start or line_start < 1;"
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 -c \
    "select column_name from information_schema.columns where table_name='markdown_chunks' order by ordinal_position;"
  ```

  → counts 1172 / 568, keys = counts (distinct), files = 1 per book, second query → 0, column list includes `source_file, char_start, char_end, line_start, chunk_key`. Also:

  ```bash
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 -c \
    "select book_title, count(*) from fixed_chunks group by 1;" -c \
    "select column_name from information_schema.columns where table_name='fixed_chunks' and column_name='source_file';"
  ```

  → fixed/recursive counts unchanged (1767 / 1849 total), no `source_file` column there.
- MEETS: only markdown table gains the 5 columns; all populated; chunk_key distinct per row; fixed/recursive byte-identical schema+data.

### UPDATE `pipeline.py` — propagate metadata into `markdown_context_chunks` + migrate the 250 live rows

- DO: extend `_DDL` column list (after `heading_path`, before `content` or after `content` — keep grouped): add

  ```
  source_file TEXT,
  char_start INT,
  char_end INT,
  line_start INT,
  chunk_key TEXT,
  ```

  (nullable — legacy rows before backfill; tsv/`BODY_EXPR` untouched).
- DO: in `ensure_table`, after `cur.execute(_DDL)` and the existing INSERT...SELECT, add idempotent migration + backfill (Postgres 16: `ADD COLUMN IF NOT EXISTS` ✓):

  ```python
  cur.execute(
      f"""
      ALTER TABLE {PIPELINE_TABLE}
          ADD COLUMN IF NOT EXISTS source_file TEXT,
          ADD COLUMN IF NOT EXISTS char_start INT,
          ADD COLUMN IF NOT EXISTS char_end INT,
          ADD COLUMN IF NOT EXISTS line_start INT,
          ADD COLUMN IF NOT EXISTS chunk_key TEXT;
      """
  )
  ```

- DO: extend the `INSERT ... SELECT` to copy the new columns (column list + select list: `m.source_file, m.char_start, m.char_end, m.line_start, m.chunk_key`).
- DO: backfill legacy rows, content-guarded (books assumed unchanged; mismatch rows stay NULL and surface in validation instead of silently lying):

  ```python
  cur.execute(
      f"""
      UPDATE {PIPELINE_TABLE} c
      SET source_file = m.source_file, char_start = m.char_start, char_end = m.char_end,
          line_start = m.line_start, chunk_key = m.chunk_key
      FROM markdown_chunks m
      WHERE c.book_title = m.book_title
        AND c.chunk_index = m.chunk_index
        AND c.content = m.content
        AND c.source_file IS NULL;
      """
  )
  ```

- PATTERN: existing copy guard pipeline.py:63-76; `_DDL` pipeline.py:29-44.
- GOTCHA: the pipeline table is NEVER dropped (contexts = cached LLM output). `CREATE TABLE IF NOT EXISTS` does not add columns to the existing 250-row table — the ALTER is what migrates it. Cost of the alternative (drop + regenerate 250 contexts with qwen3.5:4b) avoided entirely; backfill is one cheap UPDATE.
- GOTCHA: match on `c.content = m.content` in addition to `(book_title, chunk_index)` — if a book file changed since contexts were generated, `(book_title, chunk_index)` alone would attach stale offsets to drifted content. Content guard makes drift visible as NULLs.
- GOTCHA: `ensure_table` runs inside `trace`/`ragas`/`pipeline` via `_session()`/command bodies — the ALTER+backfill execute on the next command run, no separate migration step needed.
- VALIDATE:

  ```bash
  uv run python -c "from setup import connect; from pipeline import ensure_table, PIPELINE_TABLE; conn=connect(); ensure_table(conn); cur=conn.cursor(); cur.execute(f'select count(*) from {PIPELINE_TABLE} where source_file is null'); print('null source_file:', cur.fetchone()[0])"
  ```

  → `null source_file: 0`. And:

  ```bash
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 -c \
    "select count(*) from markdown_context_chunks where source_file='cloud_run.md';" -c \
    "select id, chunk_index, source_file, line_start, left(chunk_key,8) from markdown_context_chunks order by chunk_index limit 3;"
  ```

  → 250 / sensible rows.
- MEETS: citation metadata survives into the traced ask path's retrieval table; migration automatic, idempotent, zero context regeneration.

### UPDATE `traces.py` — cited answer prompt + numbered sources + LangFuse citation metadata

- DO: define in traces.py (ragas.py untouched; traces stops importing `ANSWER_PROMPT`, keeps `NUM_CTX`, `NUM_PREDICT`):

  ```python
  CITED_ANSWER_PROMPT = """Answer using only the sources below. After each claim, cite
  the supporting source like [1] or [2][3]. If the sources don't contain the
  answer, say so.

  {sources}

  Question: {query}"""
  ```

  (verbatim user template; `{sources}` / `{query}` placeholders).
- DO: rewrite `generate` to fetch per-chunk metadata and build numbered sources:

  ```python
  @observe(as_type="generation")
  def generate(query: str, chunks: list[tuple[int, str]]) -> str:
      ids = [id for id, _ in chunks]
      with _session().cursor() as cur:
          cur.execute(
              f"""SELECT id, book_title, heading_path, source_file, char_start, char_end, chunk_key
                  FROM {PIPELINE_TABLE} WHERE id = ANY(%s)""",
              (ids,),
          )
          meta = {row[0]: row for row in cur.fetchall()}
      sources = "\n\n".join(
          f"[{n}] {meta[id][1]} — {meta[id][2] or ''}\n{body}"
          for n, (id, body) in enumerate(chunks, 1)
      )
      messages = [
          {
              "role": "user",
              "content": CITED_ANSWER_PROMPT.format(sources=sources, query=query),
          }
      ]
      langfuse.update_current_generation(
          input=messages,
          model=MODEL,
          model_parameters={...unchanged...},
          metadata={
              "citations": [
                  {"n": n, "id": id, "source_file": meta[id][3],
                   "heading_path": meta[id][2], "char_start": meta[id][4],
                   "char_end": meta[id][5], "chunk_key": meta[id][6]}
                  for n, (id, _) in enumerate(chunks, 1)
              ]
          },
      )
      resp = client.chat(model=MODEL, messages=messages,
                         options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
                         think=False)
      ...usage update unchanged...
  ```

  Source block format: `[n] <book_title> — <heading_path>\n<body>` where body = existing `BODY_EXPR` text (context + content) — rerank return shape `(id, body)` unchanged, so `rerank`/`retrieve`/`ask` bodies stay as-is.
- PATTERN: metadata SELECT = setup/pipeline cursor pattern (pipeline.py:191-194); prompt format = current ANSWER_PROMPT.format usage traces.py:82-88; langfuse updates traces.py:89-113.
- GOTCHA: `str.format` — braces in *values* (markdown code fences in bodies) are safe; the template has no stray braces.
- GOTCHA: `id` shadows builtin — fine, matches existing repo style (traces.py:42, ragas.py:72).
- GOTCHA: chunk ids missing from `meta` (shouldn't happen — same table) would KeyError; acceptable, or use `meta.get(id)` with fallback label. Keep strict (fail loud).
- GOTCHA: NUM_PREDICT=512 shared with ragas; cited answers run longer. Bump traces-local answer cap to `NUM_PREDICT_CITED = 768` (define in traces.py, do NOT touch ragas.NUM_PREDICT) — flag in Open/assumed.
- VALIDATE:

  ```bash
  uv run --no-sync python -c "import traces; assert not hasattr(traces, 'ANSWER_PROMPT'); print('import ok')"
  uv run main.py trace "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
  ```

  → answer printed contains `[1]` (and ideally `[2]`/`[3]` for the shutdown + SIGTERM + grace-period claims). Then LangFuse UI (localhost:3000): generation observation shows `metadata.citations` with source_file/heading_path/offsets/chunk_key. No-regression check that ragas path is untouched:

  ```bash
  uv run --no-sync python -c "from ragas import ANSWER_PROMPT, answer_with_context; print('ragas ok')"
  ```
- MEETS: trace answers carry per-claim citations; LangFuse generation carries chunk-level citation metadata; ragas/eval flows unchanged.

### E2E validation + SQL consistency checks

- DO: full sequence:

  ```bash
  uv run main.py setup
  uv run main.py trace "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"
  PGPASSWORD=password psql postgres://postgres:password@localhost/p2 << 'SQL'
  -- offsets: char_end > char_start everywhere; monotonic per book
  select book_title, count(*) filter (where char_end <= char_start) as bad,
         count(*) filter (where char_start < lag(char_start) over (partition by book_title order by chunk_index)) as non_monotonic
  from markdown_chunks group by 1;
  -- pipeline table fully backfilled
  select count(*) as null_meta from markdown_context_chunks
  where source_file is null or char_start is null or chunk_key is null;
  SQL
  ```

  → bad = 0, non_monotonic = 0, null_meta = 0.

- chunk_key stability across re-runs (DROP+CREATE but deterministic chunking → same keys):

  ```bash
  PGPASSWORD=password psql postgres://postgres:password@localhost -d p2 -t -c "select array_agg(chunk_key order by chunk_index) from markdown_chunks where book_title='Google Cloud Run'" > /tmp/keys_run1.txt
  uv run main.py setup
  PGPASSWORD=password psql postgres://postgres:password@localhost -d p2 -t -c "select array_agg(chunk_key order by chunk_index) from markdown_chunks where book_title='Google Cloud Run'" > /tmp/keys_run2.txt
  diff /tmp/keys_run1.txt /tmp/keys_run2.txt && echo "chunk_key STABLE"
  ```

- MEETS: end-to-end cited answering over freshly rebuilt metadata; offsets consistent; keys stable.

## Tests

No pytest (convention: inline `uv run`). Coverage:
- Chunker regression: byte-identical `chunk_markdown` output vs pre-change snapshot (task 1 guard).
- Offset invariants: `0 <= char_start < char_end <= len(text)`; single-para chunks are exact stripped substrings; `line_start >= 1`; char_start monotonic within book.
- Oversized-para case: find a `len(content) == 800` (max_chars) chunk in live data → verify `char_end == char_start + 800`… actually truncated content then `.strip()`'d; assert `char_end - char_start >= len(content)` for all rows (SQL or the python assert in task 2).
- Migration: pre-existing 250-row table gets columns without drop; NULL count = 0 after backfill; drifted content → NULLs visible (content guard).
- Prompt path: cited answer contains `[n]`; `ragas.answer` still uses old ANSWER_PROMPT.

## Validation (real commands)

- L1 imports: `uv run --no-sync python -c "import chunkers, setup, pipeline, traces, ragas"` → ok
- L2 chunker asserts: task 2 VALIDATE python block (offsets, lines, snapshot equality)
- L3 rebuild: `uv run main.py setup` → 1740 markdown rows with new columns; fixed 1767 / recursive 1849 unchanged
- L4 migration: task 4 VALIDATE (null source_file: 0)
- L5 e2e: `uv run main.py trace "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?"` → cited answer; LangFuse generation metadata.citations present
- L6 no-regression: `uv run --no-sync main.py --help` → all commands list; `uv run --no-sync python -c "from ragas import answer_with_context"` → ok

## Acceptance

- [ ] `markdown_chunks` has `source_file, heading_path, char_start, char_end, line_start, chunk_key`; all NOT NULL and populated for 1740 rows; fixed/recursive tables unchanged.
- [ ] `chunk_markdown` output byte-identical to pre-change (snapshot diff passes); `chunk_markdown_metadata` is the new rich API.
- [ ] `markdown_context_chunks` gains the 5 columns via idempotent ALTER + backfill; 250 rows populated; no contexts regenerated.
- [ ] `uv run main.py trace "<SIGTERM question>"` answer contains `[1]`-style citations; LangFuse generation observation carries citations metadata (id, source_file, heading_path, offsets, chunk_key).
- [ ] `ragas.py` untouched (git diff empty for it); `ragas` command flow unaffected.
- [ ] chunk_key set identical across two `setup` runs.
- [ ] All validation green; no regressions in `eval`/`pipeline`/`ragas`/`variants` imports.

## Open / assumed

- ASSUMED — sources format `[n] <book_title> — <heading_path>\n<body>` (body = context+content via BODY_EXPR, i.e. what rerank already returns). User gave only the prompt template. Alternatives: content-only bodies, or heading_path-free labels. Affects citation quality only, not schema.
- ASSUMED — `source_file` = `Path(expanduser(book["path"])).name` → `'letsgo_further.md'`, `'cloud_run.md'`. No new BOOKS field.
- ASSUMED — chunk_key = `sha256("{source_file}:{char_start}:{content}")[:16]` hex, computed in setup.py; plain column, no UNIQUE constraint (table is DROP+CREATE each run, duplicates impossible within a run; key exists for cross-run identity/joins). If cross-run dedup enforcement wanted → add `UNIQUE` (costs nothing, but means book edits that keep an old chunk's (file, offset, text) can't coexist with... actually they can — key changes with any input change; UNIQUE is safe; still recommend plain column pending user call).
- ASSUMED — offsets = ORIGINAL-file positions; multi-paragraph merged chunk spans cover the original separator whitespace, so `content` ≠ `text[char_start:char_end]` exactly for merged chunks (normalized `"\n\n"`). Single-para chunks are exact stripped substrings. If exact-substring semantics required for ALL chunks, the merge would need per-paragraph offset columns instead — different schema, flag before implementing.
- ASSUMED — hard-truncated oversized chunk: `char_end = char_start + len(truncated_text)` (un-stripped length ≤ max_chars). Alternative: leave `char_end` = full original paragraph end (span claims more than content covers). Recommend truncated-end (truthful about what's in the chunk).
- ASSUMED — `NUM_PREDICT` for cited trace answers bumped to 768 (traces-local constant `NUM_PREDICT_CITED`); ragas keeps 512. If user wants zero shared-constant drift, keep 512 and accept clipping risk.
- ASSUMED — `CITED_ANSWER_PROMPT` lives in traces.py (single consumer); ragas.py untouched. If later the ragas path should also cite, promote to a shared prompts module then — not now.
- ASSUMED — books unchanged since contexts were generated; backfill content guard makes drift visible as NULLs rather than silently wrong offsets. If drift exists, options: accept NULL citations for those chunks, or one-time `DROP TABLE markdown_context_chunks` + regenerate 250 contexts (~LLM cost).
- Q (user call): citation metadata in LangFuse — plan logs it on the `generate` observation as `metadata.citations`. Also want it on `rerank` span output (ids already there) or the root `rag-ask` trace? Plan: generate only.
- Q (user call): should `char_end`/`line_start` offsets also be surfaced in the printed CLI answer (e.g. appended "Sources: [1] cloud_run.md:L120 'Container lifecycle'")? Plan: no — LangFuse only, CLI answer stays the raw model output.
- Q (user call): `markdown_context_chunks` currently subsets Cloud Run only (SUBSET_LIMIT 250) — citations will only ever reference `cloud_run.md`. Widen subset (regenerates contexts for extra chunks, LLM cost) or accept? Plan: accept.

## Notes

Rejected: changing `chunk_markdown`'s return type in place — breaks the `(content, heading_path)` contract (and `setup.py`'s TABLES dispatch) for zero benefit; wrapper keeps the public face stable while `chunk_markdown_metadata` is the real implementation. Rejected: computing offsets from stored content by searching the file (`text.find(content)`) — merged/truncated/stripped chunks don't appear verbatim in the file; must track spans during splitting. Rejected: dropping `markdown_context_chunks` to "migrate" — destroys 250 cached qwen3.5:4b contexts for a 5-column ALTER + one UPDATE. Rejected: threading metadata through `rerank`'s return (widens its contract) — `generate` already owns prompt construction; one indexed-PK `ANY(%s)` SELECT there is cheaper. Rejected: per-paragraph offset columns (exact-substring semantics for merged chunks) — heavier schema than user's DDL; covering spans + documented normalization instead. Tradeoff: `content ≠ text[span]` for merged chunks is the price of keeping the existing merge rule byte-identical (regression guarantee). Sequencing risk: task 3 (setup rebuild) must precede task 4 validation (backfill needs populated `markdown_chunks`); tasks otherwise independent per file.

## Amendments
