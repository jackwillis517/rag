"""DB schema + ingestion for the three chunker-variant tables.

Tables: fixed_chunks, recursive_chunks, markdown_chunks — each with a
GIN index on a generated tsvector and (post-ingestion) an HNSW index on
the embedding column. Idempotent: DROP IF EXISTS + CREATE on every run.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from chunkers import MarkdownChunk, chunk_fixed, chunk_markdown_metadata, chunk_recursive
from retrievers import _get_model

DB_URL = "postgres://postgres:password@localhost/p2"
BOOKS = [
    {"path": "~/Desktop/letsgo_further.md", "title": "Let's Go Further"},
    {"path": "~/Desktop/cloud_run.md", "title": "Google Cloud Run"},
]

TABLES = {
    "fixed_chunks": chunk_fixed,
    "recursive_chunks": chunk_recursive,
    "markdown_chunks": chunk_markdown_metadata,
}

# Citation metadata columns, markdown table only (fixed/recursive stay lean).
_MD_EXTRA = """source_file TEXT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    line_start INT NOT NULL,
    chunk_key TEXT NOT NULL,
"""

_DDL = """
DROP TABLE IF EXISTS chunks;  -- legacy single-chunker table, replaced by this restructure
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


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn


def create_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for table in TABLES:
            cur.execute(
                _DDL.format(
                    table=table, extra=_MD_EXTRA if table == "markdown_chunks" else ""
                )
            )
    conn.commit()
    print(f"Tables created: {', '.join(TABLES)} (old 'chunks' dropped if present)")


def _chunk_key(source_file: str, char_start: int, content: str) -> str:
    """Stable identity for idempotent ingestion / cross-run joins."""
    return hashlib.sha256(f"{source_file}:{char_start}:{content}".encode()).hexdigest()[:16]


def bulk_insert(
    conn: psycopg.Connection,
    table: str,
    title: str,
    rows: list[tuple[int, str, str | None]],
) -> None:
    """rows = (chunk_index, content, heading_path)."""
    if not rows:
        return
    model = _get_model()
    embeddings = model.encode([content for _, content, _ in rows], batch_size=64)
    assert embeddings.shape == (len(rows), 384)  # guard dim == VECTOR(384)
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {table} (book_title, chunk_index, content, heading_path, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (title, idx, content, path, embeddings[i])
                for i, (idx, content, path) in enumerate(rows)
            ],
        )
    conn.commit()


def bulk_insert_markdown(
    conn: psycopg.Connection,
    title: str,
    source_file: str,
    rows: list[MarkdownChunk],
) -> None:
    """Insert MarkdownChunk rows (with citation metadata) into markdown_chunks."""
    if not rows:
        return
    model = _get_model()
    embeddings = model.encode([c.content for c in rows], batch_size=64)
    assert embeddings.shape == (len(rows), 384)  # guard dim == VECTOR(384)
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO markdown_chunks
                (book_title, chunk_index, content, heading_path,
                 source_file, char_start, char_end, line_start, chunk_key, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    title,
                    i,
                    c.content,
                    c.heading_path,
                    source_file,
                    c.char_start,
                    c.char_end,
                    c.line_start,
                    _chunk_key(source_file, c.char_start, c.content),
                    embeddings[i],
                )
                for i, c in enumerate(rows)
            ],
        )
    conn.commit()


def build_hnsw(conn: psycopg.Connection, table: str) -> None:
    print(f"Building HNSW index on {table} (m=16, ef_construction=64)...")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX {table}_hnsw
            ON {table}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        cur.execute(f"ANALYZE {table};")
    conn.commit()


def verify_gin(conn: psycopg.Connection, table: str) -> None:
    """EXPLAIN (ANALYZE) a tsvector match; expect a Bitmap Index Scan on the GIN index."""
    query = "timeout | client | http"
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off;")  # tiny corpus: planner prefers Seq Scan otherwise
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM {table} "
            f"WHERE tsv @@ to_tsquery('english', '{query}')"
        )
        plan = "\n".join(row[0] for row in cur.fetchall())
    ok = f"Bitmap Index Scan on {table}_tsv_gin" in plan
    print(f"GIN verify {table}: {'PASS' if ok else 'FAIL'}")
    for line in plan.splitlines():
        print(f"    {line}")
    if not ok:
        raise RuntimeError(f"GIN index not used for {table}")


def ingest_all(conn: psycopg.Connection) -> None:
    create_tables(conn)

    for table, chunker in TABLES.items():
        for book in BOOKS:
            path = os.path.expanduser(book["path"])
            if not os.path.exists(path):
                print(f"Skipping {path} (file not found)")
                continue
            text = Path(path).read_text(encoding="utf-8")
            if chunker is chunk_markdown_metadata:
                source_file = Path(path).name  # 'letsgo_further.md' / 'cloud_run.md'
                rich = chunker(text)
                bulk_insert_markdown(conn, book["title"], source_file, rich)
                print(f"  {table}: {len(rich)} chunks for '{book['title']}'")
                continue
            chunks = chunker(text)
            rows = [(i, content, None) for i, content in enumerate(chunks)]
            bulk_insert(conn, table, book["title"], rows)
            print(f"  {table}: {len(rows)} chunks for '{book['title']}'")

    for table in TABLES:
        build_hnsw(conn, table)

    for table in TABLES:
        verify_gin(conn, table)

    print("Setup complete.")


if __name__ == "__main__":
    with connect() as conn:
        ingest_all(conn)