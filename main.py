"""Typer CLI + FastAPI server for the p2 retrieval pipeline."""

import asyncio
import json
from datetime import datetime, timezone

import ollama
import typer
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from ollama import AsyncClient
from pydantic import BaseModel

from eval import run as run_eval
from langfuse import get_client
from pipeline import (
    CONTEXT_MODEL,
    OLLAMA_HOST,
    backfill_embeddings,
    ensure_table,
    generate_contexts,
)
from pipeline import run as run_pipeline
from rag import rerank, retrieve, stream_answer
from ragas import QUESTIONS, answer_with_context, faithfulness
from setup import connect, ingest_all
from traces import ask
from variants import run as run_variants

app = typer.Typer()
fastapiapp = FastAPI()
langfuse = get_client()  # .env already loaded by the traces import above
ollama_client = AsyncClient(host=OLLAMA_HOST)
MODEL = CONTEXT_MODEL  # qwen3.5:4b


@app.command()
def setup() -> None:
    """Create the 3 chunker tables, ingest both books, build GIN/HNSW indexes."""
    with connect() as conn:
        ingest_all(conn)


@app.command(name="eval")
def eval_command() -> None:
    """Compare 5 retrievers (ts_rank_cd, BM25, semantic, RRF, RRF+cross-encoder) across chunkers; write 15-row evals.md."""
    run_eval()


@app.command()
def pipeline() -> None:
    """Evaluate the contextual-retrieval pipeline (markdown + qwen3.5:4b contexts
    + BM25 + semantic + RRF + cross-encoder rerank); builds markdown_context_chunks
    idempotently (contexts cached — second run is fast); writes pipeline.md."""
    run_pipeline()


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
        generate_contexts(conn)
        backfill_embeddings(conn)
        for question in QUESTIONS:
            answer, context = answer_with_context(client, conn, question, cache)
            score = faithfulness(client, question, answer, context)
            scores.append(score)
            typer.echo(f"  {score:.2f}  {question}  |  answer: {answer[:80]}")
    mean = sum(scores) / len(scores)
    typer.echo(f"Mean faithfulness ({len(scores)} questions): {mean:.2f}")


@app.command(name="trace")
def trace_command(question: str) -> None:
    """One LangFuse-observed RAG ask: markdown_context_chunks + RRF retrieve +
    cross-encoder rerank + qwen3.5:4b grounded answer; flushes the trace to
    Langfuse (localhost:3000)."""
    answer = ask(question)
    typer.echo(answer)
    get_client().flush()


@app.command(name="variants")
def variants_command() -> None:
    """Query-transform retrieval eval over the pipeline table (markdown_context_chunks,
    golden questions 1-10): rewrite / multi-query / HyDE vs rrf_ce baseline; LLM
    transforms via local qwen3.5:4b (temp 0, recomputed each run); appends the
    ## Query variants section to evals.md."""
    run_variants()


@app.command(name="serve")
def serve() -> None:
    """FastAPI server: POST /ask streams a cited RAG answer as SSE tokens."""
    uvicorn.run(fastapiapp, host="127.0.0.1", port=8000)


class AskRequest(BaseModel):
    query: str


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@fastapiapp.post("/ask")
async def query(body: AskRequest):
    return StreamingResponse(
        query_stream(body.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


async def query_stream(query: str):
    """SSE stream of one RAG ask: status -> token deltas (cited answer) -> done."""
    root = langfuse.start_observation(
        name="rag-ask", as_type="span", input={"query": query}
    )
    generate_observation = None
    parts: list[str] = []
    try:
        yield sse("status", {"stage": "retrieving"})

        # Sync rag.retrieve/rag.rerank go to a thread so they don't block the event loop.
        retrieve_observer = root.start_observation(
            name="retrieve", as_type="retriever", input={"query": query}
        )
        chunks = await run_in_threadpool(retrieve, query)
        retrieve_observer.update(
            output=[{"id": id, "score": score} for id, score in chunks[:10]],
            metadata={"k": len(chunks)},
        )
        retrieve_observer.end()

        rerank_observer = root.start_observation(
            name="rerank",
            as_type="span",
            input={"query": query, "n_candidates": len(chunks)},
        )
        top_chunks = await run_in_threadpool(rerank, query, chunks)
        rerank_observer.update(output=[id for id, _ in top_chunks])
        rerank_observer.end()

        yield sse("status", {"stage": "generating"})
        generate_observation = root.start_observation(
            name="generate",
            as_type="generation",
            model=MODEL,
            input={"query": query, "chunk_ids": [id for id, _ in top_chunks]},
            model_parameters={"temperature": 0},
        )
        first = True
        async for delta in stream_answer(ollama_client, query, top_chunks):
            if not delta:
                continue
            if first:  # time-to-first-token
                generate_observation.update(
                    completion_start_time=datetime.now(timezone.utc)
                )
                first = False
            parts.append(delta)
            yield sse("token", {"text": delta})

        answer = "".join(parts)
        generate_observation.update(output=answer)
        generate_observation.end()
        generate_observation = None
        root.update(output=answer)
        yield sse("done", {"answer": answer})
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected mid-stream. Record what was generated, then re-raise.
        if generate_observation is not None:
            generate_observation.update(
                output="".join(parts),
                level="WARNING",
                status_message="client disconnected",
            )
        raise
    except Exception as e:
        if generate_observation is not None:
            generate_observation.update(level="ERROR", status_message=str(e))
        yield sse("error", {"type": "stream_error", "message": "generation failed"})
    finally:
        if generate_observation is not None:
            generate_observation.end()
        root.end()


if __name__ == "__main__":
    app()
