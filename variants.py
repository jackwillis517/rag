"""Query-transform retrievers: rewrite / multi-query / HyDE over the rrf_ce base
(markdown_context_chunks + BM25/semantic RRF + cross-encoder), evaluated on
golden questions 1-10; LLM = local qwen3.5:4b via ragas._chat.

Multi-query: original + LLM variants each through the RRF leg (top-10), union
candidates, cross-encoder rerank anchored on the ORIGINAL question (D3).
HyDE: hypothetical document embedding replaces the query embedding in the
semantic leg ONLY; BM25 leg + cross-encoder keep the original question (D5).
Rewrite: rewritten query feeds the entire base rrf_ce retriever (D7).
"""

import re
from collections.abc import Callable

import ollama
from ranx import Run, fuse

from eval import EVALS_PATH, evaluate, load_golden, relevant_ids
from pipeline import (
    BODY_EXPR,
    OLLAMA_HOST,
    PIPELINE_TABLE,
    _connectivity,
    backfill_embeddings,
    ensure_table,
    generate_contexts,
)
from ragas import _chat
from retrievers import (
    CANDIDATES,
    K,
    RRF_CE_CANDIDATES,
    RRF_K,
    _bm25_corpus,
    _bm25_ranked,
    _cross_encoder_rerank,
    _get_cross_encoder,
    _get_model,
    _rrf_ranked,
    _semantic_ranked,
    rrf_ce_retriever,
)
from setup import connect

N_VARIANTS = 3  # multi-query legs generated per question (plus the original)

VARIANTS_HEADER = "## Query variants"  # eval.py preserves everything from here down

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


def _clean_question(line: str) -> str:
    line = line.strip().lstrip("-*• ").strip()
    return re.sub(r"^(?:\d+[.:)])?\s*", "", line).strip()


def parse_multiquery(out: str) -> list[str]:
    """Up to 3 distinct non-empty question lines, bullets/numbering stripped."""
    qs: list[str] = []
    seen: set[str] = set()
    for line in out.splitlines():
        q = _clean_question(line)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            qs.append(q)
    return qs[:N_VARIANTS]


def parse_rewrite(out: str, question: str) -> str:
    """Rewritten query; empty LLM output -> original question (leg must run)."""
    return out.strip() or question


def parse_hyde(out: str, question: str) -> str:
    """Hypothetical document for the semantic leg; empty -> original question."""
    return out.strip() or question


def rewrite_rrf_ce_retriever(
    conn, table: str, cache: dict, client: ollama.Client, body_expr: str = "content"
) -> Callable[[str], list[int]]:
    """Question -> top-5 ids: LLM-rewritten query through the whole base rrf_ce."""
    base = rrf_ce_retriever(conn, table, cache, body_expr)

    def retrieve(question: str, k: int = K) -> list[int]:
        rewritten = parse_rewrite(_chat(client, REWRITE_PROMPT.format(question=question)), question)
        return base(rewritten)

    return retrieve


def multiquery_rrf_ce_retriever(
    conn, table: str, cache: dict, client: ollama.Client, body_expr: str = "content"
) -> Callable[[str], list[int]]:
    """Question -> top-5 ids: original + variants, per-leg RRF top-10 -> union ->
    cross-encoder rerank anchored on the ORIGINAL question."""
    ranked = _rrf_ranked(conn, table, cache, body_expr)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()

    def retrieve(question: str, k: int = K) -> list[int]:
        legs = [question] + parse_multiquery(
            _chat(client, MULTIQUERY_PROMPT.format(question=question))
        )
        union: dict[int, str] = {}  # id -> body, first-seen order
        for leg in legs:
            for id, _ in ranked(leg, RRF_CE_CANDIDATES):
                union.setdefault(id, body_of[id])
        if not union:
            return []
        return _cross_encoder_rerank(ce, question, list(union.items()), k)

    return retrieve


def hyde_rrf_ce_retriever(
    conn, table: str, cache: dict, client: ollama.Client, body_expr: str = "content"
) -> Callable[[str], list[int]]:
    """Question -> top-5 ids: hypothetical doc embeds in the semantic leg only
    (BM25 + cross-encoder see the original question); RRF fuse -> CE rerank."""
    sem = _semantic_ranked(conn, table, _get_model())
    bm = _bm25_ranked(conn, table, cache, body_expr)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()

    def retrieve(question: str, k: int = K) -> list[int]:
        doc = parse_hyde(_chat(client, HYDE_PROMPT.format(question=question)), question)
        sem_list = sem(doc, CANDIDATES)
        bm_list = bm(question, CANDIDATES)
        # fusion block copied from _rrf_ranked (retrievers.py:158-169), guards included
        if not sem_list:
            fused = [(id, score) for id, score in bm_list[:RRF_CE_CANDIDATES]]
        elif not bm_list:
            fused = [(id, sim) for id, sim in sem_list[:RRF_CE_CANDIDATES]]
        else:
            run_sem = Run.from_dict({"q": {str(id): sim for id, sim in sem_list}})
            run_bm = Run.from_dict({"q": {str(id): score for id, score in bm_list}})
            fused = fuse(runs=[run_sem, run_bm], norm=None, method="rrf", params={"k": RRF_K})
            order = sorted(fused.to_dict()["q"].items(), key=lambda kv: (-kv[1], int(kv[0])))
            fused = [(int(doc_id), float(score)) for doc_id, score in order[:RRF_CE_CANDIDATES]]
        if not fused:
            return []
        cands = [(id, body_of[id]) for id, _ in fused]
        return _cross_encoder_rerank(ce, question, cands, k)

    return retrieve


def load_variants_section() -> str:
    """Current variants section of evals.md ('' if absent) — eval.py re-appends it."""
    if not EVALS_PATH.exists():
        return ""
    text = EVALS_PATH.read_text()
    i = text.find(VARIANTS_HEADER)
    return text[i:] if i >= 0 else ""


def run() -> None:
    """Evaluate rewrite / multiquery / hyde vs plain rrf_ce on golden ids 1-10."""
    _connectivity()
    client = ollama.Client(host=OLLAMA_HOST)
    golden = [q for q in load_golden() if q["id"] <= 10]
    cache: dict = {}
    with connect() as conn:
        ensure_table(conn)
        generate_contexts(conn)
        backfill_embeddings(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, content FROM {PIPELINE_TABLE}")  # content ONLY
            rows = cur.fetchall()
        relevant = [relevant_ids(rows, q["ground_truth_snippet"]) for q in golden]
        retrievers = [
            ("rrf_ce (baseline)", rrf_ce_retriever(conn, PIPELINE_TABLE, cache, body_expr=BODY_EXPR)),
            ("rewrite", rewrite_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
            ("multiquery", multiquery_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
            ("hyde", hyde_rrf_ce_retriever(conn, PIPELINE_TABLE, cache, client, body_expr=BODY_EXPR)),
        ]
        results = [(name, evaluate(fn, golden, relevant)) for name, fn in retrievers]

    lines = [
        f"{VARIANTS_HEADER}: rrf_ce base (markdown_context + BM25/semantic RRF + "
        f"cross-encoder), k=5, golden questions 1-10 (Cloud Run Runtime)",
        "",
        "| Retriever | recall@5 | MRR | MAP |",
        "|---|---|---|---|",
        *[f"| {n} | {m['recall']:.2f} | {m['mrr']:.2f} | {m['map']:.2f} |" for n, m in results],
        "",
        "_NOT comparable to the 50-question averages above — 10-question subset, contextual table only._",
    ]
    prefix = ""
    if EVALS_PATH.exists():
        prefix = EVALS_PATH.read_text().split(VARIANTS_HEADER)[0].rstrip("\n")
    EVALS_PATH.write_text(prefix + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Updated {EVALS_PATH}")


if __name__ == "__main__":
    run()