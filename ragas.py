"""RAGAS-style faithfulness: qwen3.5:4b answers grounded in retrieved chunks;
statement extraction + verification; F = supported/total; reference-free."""

import re

import httpx
import ollama

from pipeline import BODY_EXPR, CONTEXT_MODEL, OLLAMA_HOST, PIPELINE_TABLE
from retrievers import _bm25_corpus, rrf_ce_retriever

NUM_CTX = 8192  # 5 chunks (~2k tokens) + prompts; far under qwen3.5:4b max 262144
NUM_PREDICT = 512  # cap; verdict/extraction outputs are shorter — max, not target

QUESTIONS: list[str] = [
    "How much time does a Cloud Run container get to shut down cleanly after receiving SIGTERM?",
    "What does setting concurrency to 1 mean for a Cloud Run service?",
    "Is the Cloud Run container filesystem backed by disk, and how large can it get?",
    "How does a multistage Dockerfile let you ship a small image with only the compiled binary?",
    "How does the Cloud SQL Proxy Server authenticate incoming connections?",
]

ANSWER_PROMPT = (
    "Answer the question using only the information in the provided context. "
    "If the context does not contain the answer, say so.\n"
    "context:\n{context}\n"
    "question: {question}"
)

EXTRACT_PROMPT = (
    "Given a question and answer, create one or more statements from each sentence "
    "in the given answer.\n"
    "question: {question}\n"
    "answer: {answer}"
)

VERIFY_PROMPT = (
    "For each numbered statement, output true if supported by context, false otherwise. "
    "Output exactly one boolean per statement, in order.\n"
    "statements:\n{statements}\n"
    "context:\n{context}"
)


def _chat(client: ollama.Client, prompt: str) -> str:
    """One chat call -> stripped text. Single retry on transient errors."""
    for attempt in (0, 1):
        try:
            resp = client.chat(
                model=CONTEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
                think=False,
            )
            return resp.message.content.strip()
        except (ollama.ResponseError, httpx.HTTPError):
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


def answer(client: ollama.Client, question: str, context: str) -> str:
    """LLM answer grounded in the retrieved context."""
    return _chat(client, ANSWER_PROMPT.format(context=context, question=question))


def answer_with_context(
    client: ollama.Client, conn, question: str, cache: dict
) -> tuple[str, str]:
    """Retrieve top-5 chunks, answer from their combined bodies -> (answer, context)."""
    retriever = rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)
    ids = retriever(question)
    ids_all, bodies = _bm25_corpus(conn, PIPELINE_TABLE, cache, BODY_EXPR)
    body_of = dict(zip(ids_all, bodies))
    context = "\n\n".join(body_of[i] for i in ids)
    return answer(client, question, context), context


def _clean_line(line: str) -> str:
    """Strip list markers / leading numbers, keep the statement text."""
    line = line.strip().lstrip("-* ").strip()
    return re.sub(r"^(?:statement\s*)?\d+[.:)]\s*", "", line, flags=re.IGNORECASE).strip()


def _statements(client: ollama.Client, question: str, answer: str) -> list[str]:
    """Decompose the answer into a list of short factual statements."""
    prompt = EXTRACT_PROMPT.format(question=question, answer=answer)
    out = _chat(client, prompt)
    return [s for s in (_clean_line(line) for line in out.splitlines()) if s]


def _verdicts(client: ollama.Client, statements: list[str], context: str) -> list[bool]:
    """One boolean per statement (in order): supported by the context or not."""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    prompt = VERIFY_PROMPT.format(statements=numbered, context=context)
    tokens = re.findall(r"\b(true|false|yes|no)\b", _chat(client, prompt), flags=re.IGNORECASE)
    verdicts = [t.lower() in ("true", "yes") for t in tokens]
    return (verdicts + [False] * len(statements))[: len(statements)]


def faithfulness(client: ollama.Client, question: str, answer: str, context: str) -> float:
    """F = supported statements / total statements (1.0 if no statements)."""
    if not answer.strip():
        return 1.0
    statements = _statements(client, question, answer)
    if not statements:
        return 1.0
    verdicts = _verdicts(client, statements, context)
    return sum(verdicts) / len(statements)