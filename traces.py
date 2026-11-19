"""LangFuse-observed RAG ask: hybrid RRF retrieve -> cross-encoder rerank -> qwen3.5:4b grounded answer."""

from dotenv import load_dotenv

load_dotenv()  # .env holds LANGFUSE_* keys; uv run does not load it

import ollama  # noqa: E402

from langfuse import get_client, observe  # noqa: E402
from pipeline import (  # noqa: E402
    BODY_EXPR,
    CONTEXT_MODEL,
    OLLAMA_HOST,
    PIPELINE_TABLE,
    backfill_embeddings,
    ensure_table,
    generate_contexts,
)
from ragas import NUM_CTX  # noqa: E402
from retrievers import (  # noqa: E402
    CANDIDATES,
    K,
    _bm25_corpus,
    _cross_encoder_rerank,
    _get_cross_encoder,
    _rrf_ranked,
)
from setup import connect  # noqa: E402

langfuse = get_client()
client = ollama.Client(host=OLLAMA_HOST)
MODEL = CONTEXT_MODEL  # qwen3.5:4b
NUM_PREDICT_CITED = 768  # cited answers run longer than ragas' 512 cap

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

_conn = None  # module-lazy psycopg connection
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


@observe(as_type="retriever")
def retrieve(query: str, k: int = CANDIDATES) -> list[tuple[int, float]]:
    """Hybrid RRF retrieval (semantic + BM25 fusion) over the pipeline table."""
    ranked = _rrf_ranked(_session(), PIPELINE_TABLE, _cache, body_expr=BODY_EXPR)
    hits = ranked(query, k)
    langfuse.update_current_span(
        input={"query": query, "k": k},
        output=[{"id": id, "score": score} for id, score in hits[:10]],
        metadata={"k": k, "n_hits": len(hits)},
    )
    return hits


@observe()
def rerank(
    query: str, hits: list[tuple[int, float]], top_n: int = K
) -> list[tuple[int, str]]:
    """Cross-encoder rerank of the fused candidates -> top_n (id, body) pairs."""
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


@observe(as_type="generation")
def generate(query: str, chunks: list[tuple[int, str]]) -> str:
    """Grounded, cited answer from the retrieved chunk bodies via qwen3.5:4b."""
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
    messages = [
        {
            "role": "user",
            "content": CITED_ANSWER_PROMPT.format(sources=sources, query=query),
        }
    ]
    langfuse.update_current_generation(
        input=messages,
        model=MODEL,
        model_parameters={
            "temperature": 0,
            "num_predict": NUM_PREDICT_CITED,
            "num_ctx": NUM_CTX,
        },
        metadata={
            "citations": [
                {
                    "n": n,
                    "id": id,
                    "source_file": meta[id][3],
                    "heading_path": meta[id][2],
                    "char_start": meta[id][4],
                    "char_end": meta[id][5],
                    "line_start": meta[id][6],
                    "chunk_key": meta[id][7],
                }
                for n, (id, _) in enumerate(chunks, 1)
            ],
        },
    )
    resp = client.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 0,
            "num_predict": NUM_PREDICT_CITED,
            "num_ctx": NUM_CTX,
        },
        think=False,
    )
    text = resp.message.content.strip()
    langfuse.update_current_generation(
        output=text,
        usage_details={
            "input": resp.prompt_eval_count or 0,
            "output": resp.eval_count or 0,
        },
    )
    return text


@observe(name="rag-ask")
def ask(query: str) -> str:
    """Root trace: retrieve -> rerank -> generate."""
    return generate(query, rerank(query, retrieve(query)))

