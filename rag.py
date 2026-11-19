"""Tracing-free RAG ask: hybrid RRF retrieve -> cross-encoder rerank -> async streaming cited answer (qwen3.5:4b)."""

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

MODEL = CONTEXT_MODEL  # qwen3.5:4b
NUM_PREDICT_CITED = 768  # cited answers run longer than ragas' 512 cap

# verbatim copy of traces.CITED_ANSWER_PROMPT — kept separate so rag.py imports no
# langfuse; keep in sync (validation ladder asserts equality)
CITED_ANSWER_PROMPT = """Answer using only the sources below. After each claim, cite the
supporting source like [1] or [2][3]. If the sources don't contain the answer,
say so.

End your response with a "Citations:" section listing every cited source once,
in citation order, one per line, in the format:
<number>. [<citation>] - <source> - <other metadata>
where <source> is the source's book (file) and <other metadata> is its heading
path and line number. Example:

Citations:
1. [1] - Google Cloud Run (cloud_run.md) - Container lifecycle, line 123

{sources}

Question: {query}"""

_conn = None  # module-lazy psycopg connection (mirrors traces.py, minus tracing)
_cache: dict = {}  # shared BM25/embedding corpus cache (retrievers pattern)


def _session():
    """Lazy idempotent session: connect -> ensure_table -> contexts -> embeddings."""
    global _conn
    if _conn is None:
        _conn = connect()
        ensure_table(_conn)
        generate_contexts(_conn)
        backfill_embeddings(_conn)
    return _conn


def retrieve(query: str, k: int = CANDIDATES) -> list[tuple[int, float]]:
    """Hybrid RRF retrieval (semantic + BM25 fusion) over the pipeline table."""
    ranked = _rrf_ranked(_session(), PIPELINE_TABLE, _cache, body_expr=BODY_EXPR)
    hits = ranked(query, k)
    return hits


def rerank(
    query: str, hits: list[tuple[int, float]], top_n: int = K
) -> list[tuple[int, str]]:
    """Cross-encoder rerank of the fused candidates -> top_n (id, body) pairs."""
    ids, bodies = _bm25_corpus(_session(), PIPELINE_TABLE, _cache, BODY_EXPR)
    body_of = dict(zip(ids, bodies))
    candidates = [(id, body_of[id]) for id, _ in hits]
    top_ids = _cross_encoder_rerank(_get_cross_encoder(), query, candidates, top_n)
    top = [(id, body_of[id]) for id in top_ids]
    return top


def _sources_block(chunks: list[tuple[int, str]]) -> str:
    """Numbered sources block with per-chunk citation metadata."""
    ids = [id for id, _ in chunks]
    with _session().cursor() as cur:
        cur.execute(
            f"""SELECT id, book_title, heading_path, source_file, char_start, char_end, line_start, chunk_key
                FROM {PIPELINE_TABLE} WHERE id = ANY(%s)""",
            (ids,),
        )
        meta = {row[0]: row for row in cur.fetchall()}
    return "\n\n".join(
        f"[{n}] {meta[id][1]} — {meta[id][2] or ''} ({meta[id][3]}, line {meta[id][6]})\n{body}"
        for n, (id, body) in enumerate(chunks, 1)
    )


async def stream_answer(
    client: ollama.AsyncClient, query: str, chunks: list[tuple[int, str]]
) -> AsyncIterator[str]:
    """Async streaming cited answer; yields content deltas (qwen3.5:4b).

    Consumed as:
        chunks = rerank(query, retrieve(query))   # sync, heavy (CE model load)
        async for delta in stream_answer(client, query, chunks): ...
    Generator body is lazy — the first iteration pays _sources_block's DB SELECT
    (plus one-time _session init: ensure_table/contexts/embeddings, all cached).
    No retry inside: a partially consumed stream cannot be retried without
    duplicating yielded output; ollama/httpx errors propagate to the consumer.
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

