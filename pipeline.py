"""Contextual-retrieval pipeline: markdown chunks + qwen3.5:4b contexts
+ BM25 + semantic + RRF fusion + cross-encoder rerank, evaluated on the golden set.

Contextual retrieval (Anthropic recipe): before indexing, each chunk gets a
short LLM-generated context situating it in the whole document; context is
prepended for BM25 corpus, embeddings, and rerank passages. Original chunk
text stays in `content` so snippet-containment relevance is unaffected.

The `markdown_context_chunks` table is built and owned by this module (not
setup.py) and is never dropped: contexts are expensive LLM output, cached
across runs. SUBSET_BOOK/SUBSET_LIMIT restrict the corpus (and thus LLM calls)
to a much smaller slice of one book; golden questions whose snippet matches
no subset chunk are skipped by the eval (counted in pipeline.md). If the
source books or subset constants change, drop the table manually and
re-run `pipeline`.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import ollama
import typer

from eval import K, evaluate, load_golden, relevant_ids
from retrievers import _get_model, rrf_ce_retriever
from setup import BOOKS, connect

OLLAMA_HOST = "http://localhost:11434"  # ollama-python default
CONTEXT_MODEL = "qwen3.5:4b"
CONTEXT_NUM_CTX = 262144  # qwen3.5:4b max_position_embeddings (verified)
CONTEXT_WORKERS = 1  # ThreadPoolExecutor max_workers (VRAM-tight: 1 slot; see report)  # noqa: E501
CONTEXT_MAX_PREDICT = 128  # cap output; contexts are short
CONTEXT_COMMIT_BATCH = 32  # UPDATE commit granularity (crash-safe cache)
PIPELINE_TABLE = "markdown_context_chunks"
SUBSET_BOOK = "Google Cloud Run"  # subset: one book (smaller doc prefill, fewer chunks)
SUBSET_LIMIT = 250  # contiguous chunks from the book start (chunk_index < limit)
# D1 SQL concatenation: context + newline + content (matches _combined below)
BODY_EXPR = "coalesce(context, '') || E'\\n' || content"
PIPELINE_MD = Path("pipeline.md")

PROMPT_TEMPLATE = """<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

_DDL = f"""
CREATE TABLE IF NOT EXISTS {PIPELINE_TABLE} (
    id SERIAL PRIMARY KEY,
    book_title TEXT NOT NULL,
    chunk_index INT NOT NULL,
    heading_path TEXT,
    source_file TEXT,               -- citation metadata (nullable: legacy rows pre-backfill)
    char_start INT,
    char_end INT,
    line_start INT,
    chunk_key TEXT,
    content TEXT NOT NULL,          -- ORIGINAL chunk text (relevance containment)
    context TEXT,                   -- LLM context; NULL until generated
    embedding VECTOR(384),          -- NULL until backfilled over context+content
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
        setweight(to_tsvector('english', {BODY_EXPR}), 'B')
    ) STORED
);
CREATE INDEX IF NOT EXISTS {PIPELINE_TABLE}_tsv_gin
    ON {PIPELINE_TABLE} USING GIN (tsv);
"""


def _combined(context: str | None, content: str) -> str:
    """Canonical Python concatenation for embeddings / CE passages (D1)."""
    return f"{context}\n{content}"


def _connectivity() -> None:
    """Exit cleanly if ollama is down or the model is not pulled."""
    try:
        resp = ollama.list()
    except (httpx.ConnectError, ollama.ResponseError):
        typer.echo(
            f"ollama not reachable at {OLLAMA_HOST} — start it (ollama serve) and pull "
            f"the model (ollama pull {CONTEXT_MODEL})",
            err=True,
        )
        raise typer.Exit(1)
    if CONTEXT_MODEL not in [m.model for m in resp.models]:
        typer.echo(f"model {CONTEXT_MODEL} not pulled — run: ollama pull {CONTEXT_MODEL}", err=True)
        raise typer.Exit(1)


def _doc_texts() -> dict[str, str]:
    """{book_title: full markdown text} from the setup.BOOKS paths."""
    docs = {}
    for book in BOOKS:
        path = os.path.expanduser(book["path"])
        if not os.path.exists(path):
            typer.echo(f"Book file not found: {path}", err=True)
            raise typer.Exit(1)
        docs[book["title"]] = Path(path).read_text(encoding="utf-8")
    return docs


def ensure_table(conn) -> None:
    """Create the contextual table idempotently and copy any missing subset chunks.

    Citation metadata columns are added (IF NOT EXISTS) and copied from
    markdown_chunks; the legacy 250 pre-metadata rows are backfilled in-place
    (content-guarded — drifted books surface as NULLs, never stale offsets).
    """
    with conn.cursor() as cur:
        cur.execute(_DDL)
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
        cur.execute(
            f"""
            INSERT INTO {PIPELINE_TABLE}
                (book_title, chunk_index, heading_path,
                 source_file, char_start, char_end, line_start, chunk_key, content)
            SELECT m.book_title, m.chunk_index, m.heading_path,
                   m.source_file, m.char_start, m.char_end, m.line_start, m.chunk_key,
                   m.content
            FROM markdown_chunks m
            WHERE m.book_title = %s AND m.chunk_index < %s
              AND NOT EXISTS (
                SELECT 1 FROM {PIPELINE_TABLE} c
                WHERE c.book_title = m.book_title AND c.chunk_index = m.chunk_index
            );
            """,
            (SUBSET_BOOK, SUBSET_LIMIT),
        )
        cur.execute(
            f"""
            UPDATE {PIPELINE_TABLE} c
            SET source_file = m.source_file, char_start = m.char_start,
                char_end = m.char_end, line_start = m.line_start, chunk_key = m.chunk_key
            FROM markdown_chunks m
            WHERE c.book_title = m.book_title
              AND c.chunk_index = m.chunk_index
              AND c.content = m.content
              AND c.source_file IS NULL;
            """
        )
    conn.commit()


def _one_context(client, doc_text: str, chunk: str, row_id: int) -> tuple[int, str]:
    """One LLM call -> (row_id, context). Single retry on transient errors."""
    prompt = (
        PROMPT_TEMPLATE.replace("{{WHOLE_DOCUMENT}}", doc_text).replace("{{CHUNK_CONTENT}}", chunk)
    )
    for attempt in (0, 1):
        try:
            resp = client.chat(
                model=CONTEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0,
                    "num_predict": CONTEXT_MAX_PREDICT,
                    "num_ctx": CONTEXT_NUM_CTX,
                },
                think=False,
            )
            return row_id, resp.message.content.strip()
        except (ollama.ResponseError, httpx.HTTPError):
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


def generate_contexts(conn) -> None:
    """Fill the `context` column for every NULL row (resumable; batched commits)."""
    _connectivity()
    docs = _doc_texts()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, book_title, content FROM {PIPELINE_TABLE} "
            f"WHERE context IS NULL ORDER BY book_title, chunk_index"
        )
        rows = cur.fetchall()
    total = len(rows)
    if not total:
        print("All contexts already generated (cached).")
        return
    print(f"Generating {total} contexts with {CONTEXT_MODEL} ({CONTEXT_WORKERS} workers)...")
    client = ollama.Client(host=OLLAMA_HOST)
    t0 = time.time()
    done = 0
    buffer: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=CONTEXT_WORKERS) as pool:
        futures = {
            pool.submit(_one_context, client, docs[row[1]], row[2], row[0]): row[0] for row in rows
        }
        for fut in as_completed(futures):
            row_id, context = fut.result()
            buffer.append((context, row_id))
            done += 1
            if len(buffer) >= CONTEXT_COMMIT_BATCH:
                _commit_contexts(conn, buffer)
                buffer = []
            if done % 100 == 0 or done == total:
                rate = done / (time.time() - t0)
                print(f"  {done}/{total} contexts ({rate:.1f}/s)")
    if buffer:
        _commit_contexts(conn, buffer)


def _commit_contexts(conn, buffer: list[tuple[str, int]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            f"UPDATE {PIPELINE_TABLE} SET context = %s WHERE id = %s", buffer
        )
    conn.commit()


def backfill_embeddings(conn) -> None:
    """Embed context+content for rows with NULL embedding; then build HNSW."""
    model = _get_model()  # same 384-dim bi-encoder as setup.bulk_insert
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM {PIPELINE_TABLE} WHERE embedding IS NULL"
        )
        pending = cur.fetchone()[0]
    if pending:
        print(f"Backfilling {pending} embeddings over context+content...")
        batch_size = 512
        for offset in range(0, pending, batch_size):
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, context, content FROM {PIPELINE_TABLE} "
                    f"WHERE embedding IS NULL ORDER BY id LIMIT {batch_size}"
                )
                rows = cur.fetchall()
            texts = [_combined(context, content) for _, context, content in rows]
            embeddings = model.encode(texts, batch_size=64)
            with conn.cursor() as cur:
                cur.executemany(
                    f"UPDATE {PIPELINE_TABLE} SET embedding = %s WHERE id = %s",
                    [(embeddings[i], rows[i][0]) for i in range(len(rows))],
                )
            conn.commit()
    else:
        print("All embeddings already backfilled (cached).")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {PIPELINE_TABLE}_hnsw
            ON {PIPELINE_TABLE}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        cur.execute(f"ANALYZE {PIPELINE_TABLE};")
    conn.commit()


def run() -> None:
    golden = load_golden()
    cache: dict = {}
    with connect() as conn:
        ensure_table(conn)
        generate_contexts(conn)
        backfill_embeddings(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, content FROM {PIPELINE_TABLE}")  # content ONLY
            rows = cur.fetchall()
        relevant = [relevant_ids(rows, q["ground_truth_snippet"]) for q in golden]
        retriever = rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)
        m = evaluate(retriever, golden, relevant)  # k=K=5
    skipped = sum(1 for r in relevant if not r)

    lines = [
        f"# Pipeline result: contextual retrieval + rrf_ce "
        f"(k={K}, subset {SUBSET_BOOK} chunks 0-{SUBSET_LIMIT - 1}, "
        f"{m['n']}/{len(golden)} questions evaluated)",
        "",
        "| Retriever | Chunker | recall@5 | MRR | MAP |",
        "|---|---|---|---|---|",
        f"| rrf_ce | markdown_context | {m['recall']:.2f} | {m['mrr']:.2f} | {m['map']:.2f} |",
        "",
        f"_Skipped (no snippet match in subset): markdown_context {skipped}_",
    ]
    out = "\n".join(lines) + "\n"
    PIPELINE_MD.write_text(out)
    print(out)
    print(f"Wrote {PIPELINE_MD}")


if __name__ == "__main__":
    run()
