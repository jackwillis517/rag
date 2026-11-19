"""Retrievers: Postgres FTS, BM25, pgvector semantic, ranx RRF fusion.

All public functions are factories returning question -> top-k chunk-id
closures, mirroring the original bm25_retriever pattern (eval.py:45).
"""

import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi
from ranx import Run, fuse

K = 5  # eval depth, same as eval.K
RRF_K = 60  # ranx RRF constant (Cormack et al. default)
CANDIDATES = 50  # per-retriever depth fed INTO fusion (>= K)
RRF_CE_CANDIDATES = 10  # RRF out-of-fusion depth feeding the cross-encoder
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, matches VECTOR(384)
CROSS_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L12-v2"  # BERT, 512 max tokens, ~127MB
)

_MODEL = None
_CROSS = None


def _get_model():
    """Lazy singleton: module import stays cheap; setup.py reuses this for ingestion."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _get_cross_encoder():
    """Lazy singleton, same pattern as _get_model."""
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder

        _CROSS = CrossEncoder(CROSS_MODEL_NAME)
    return _CROSS


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def ts_retriever(conn, table: str) -> Callable[[str], list[int]]:
    """Top-k chunk ids by ts_rank_cd over the generated tsvector.

    Fair baseline: websearch_to_tsquery ANDs every question term, so most
    natural-language questions match zero chunks. Instead OR the question
    tokens — to_tsquery stems each token and drops stopwords via the
    english config — and let ts_rank_cd sort the matches.
    """

    def retrieve(question: str, k: int = K) -> list[int]:
        terms = " | ".join(re.findall(r"[a-z0-9]+", question.lower()))
        if not terms:
            return []
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id FROM {table}
                WHERE tsv @@ to_tsquery('english', %s)
                ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (terms, terms, k),
            )
            return [row[0] for row in cur.fetchall()]

    return retrieve


def _bm25_corpus(
    conn, table: str, cache: dict, body_expr: str = "content"
) -> tuple[list[int], list[str]]:
    """(ids, bodies) for the table's selected body; triggers the shared cache entry."""
    if table not in cache:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, {body_expr} FROM {table}")
            rows = cur.fetchall()
        ids = [row[0] for row in rows]
        bodies = [row[1] for row in rows]
        cache[table] = (ids, bodies, BM25Okapi([_tokenize(b) for b in bodies]))
    ids, bodies, _ = cache[table]
    return ids, bodies


def _bm25_ranked(
    conn, table: str, cache: dict, body_expr: str = "content"
) -> Callable[[str, int], list[tuple[int, float]]]:
    """Build the BM25 index for a table once (cached), return question -> (id, score) list."""
    _bm25_corpus(conn, table, cache, body_expr)
    ids, _, bm25 = cache[table]

    def ranked(question: str, depth: int) -> list[tuple[int, float]]:
        scores = bm25.get_scores(_tokenize(question))
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:depth]
        return [(ids[i], float(scores[i])) for i in order]

    return ranked


def bm25_retriever(conn, table: str, cache: dict) -> Callable[[str], list[int]]:
    """Question -> top-k chunk ids by BM25Okapi over the table corpus."""
    ranked = _bm25_ranked(conn, table, cache)

    def retrieve(question: str, k: int = K) -> list[int]:
        return [id for id, _ in ranked(question, k)]

    return retrieve


def _semantic_ranked(
    conn, table: str, model
) -> Callable[[str, int], list[tuple[int, float]]]:
    """Question -> (id, cosine_similarity) list via pgvector `<=>` cosine distance."""

    def ranked(question: str, depth: int) -> list[tuple[int, float]]:
        qvec = model.encode(question)  # np.ndarray shape (384,)
        with conn.cursor() as cur:
            # default ef_search=40 < CANDIDATES=50 -> widen the approximate scan
            cur.execute(f"SET hnsw.ef_search = {CANDIDATES * 2};")
            cur.execute(
                f"""
                SELECT id, 1 - (embedding <=> %s) AS sim FROM {table}
                ORDER BY embedding <=> %s, id ASC
                LIMIT %s
                """,
                (qvec, qvec, depth),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    return ranked


def semantic_search_retriever(conn, table: str) -> Callable[[str], list[int]]:
    """Question -> top-k chunk ids by pgvector cosine similarity."""
    ranked = _semantic_ranked(conn, table, _get_model())

    def retrieve(question: str, k: int = K) -> list[int]:
        return [id for id, _ in ranked(question, k)]

    return retrieve


def _rrf_ranked(
    conn, table: str, cache: dict, body_expr: str = "content"
) -> Callable[[str, int], list[tuple[int, float]]]:
    """Question, depth -> [(id, fused_score)] by RRF fusion of semantic + BM25 (ranx)."""
    sem = _semantic_ranked(conn, table, _get_model())
    bm = _bm25_ranked(conn, table, cache, body_expr)

    def ranked(question: str, depth: int) -> list[tuple[int, float]]:
        sem_list = sem(question, CANDIDATES)
        bm_list = bm(question, CANDIDATES)
        # Run.from_dict crashes on empty inner dicts -> fall back to the non-empty run
        if not sem_list:
            return [(id, score) for id, score in bm_list[:depth]]
        if not bm_list:
            return [(id, sim) for id, sim in sem_list[:depth]]
        run_sem = Run.from_dict({"q": {str(id): sim for id, sim in sem_list}})
        run_bm = Run.from_dict({"q": {str(id): score for id, score in bm_list}})
        fused = fuse(
            runs=[run_sem, run_bm], norm=None, method="rrf", params={"k": RRF_K}
        )
        order = sorted(
            fused.to_dict()["q"].items(), key=lambda kv: (-kv[1], int(kv[0]))
        )
        return [(int(doc_id), fused_score) for doc_id, fused_score in order[:depth]]

    return ranked


def rrf_retriever(conn, table: str, cache: dict) -> Callable[[str], list[int]]:
    """Question -> top-k chunk ids by RRF fusion of semantic + BM25 (ranx)."""
    ranked = _rrf_ranked(conn, table, cache)

    def retrieve(question: str, k: int = K) -> list[int]:
        return [id for id, _ in ranked(question, k)]

    return retrieve


def _cross_encoder_rerank(
    ce, question: str, candidates: list[tuple[int, str]], k: int
) -> list[int]:
    """Top-k ids by cross-encoder score over (question, body) pairs.

    `predict` returns RAW LOGITS (Identity activation; can be negative) —
    monotonic, so ranking is unaffected. Tie-break on candidate id for determinism.
    """
    pairs = [(question, body) for _, body in candidates]
    scores = ce.predict(pairs, batch_size=32)
    order = sorted(
        range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i][0])
    )[:k]
    return [candidates[i][0] for i in order]


def rrf_ce_retriever(
    conn, table: str, cache: dict, body_expr: str = "content"
) -> Callable[[str], list[int]]:
    """Question -> top-5 ids: RRF (semantic + BM25) top-10 -> cross-encoder rerank -> top-k.

    RRF + Cross Encoder Rerank: reranking happens on the fused candidate pool
    (depth RRF_CE_CANDIDATES, out of fusion), not on a single retriever.
    """
    ranked = _rrf_ranked(conn, table, cache, body_expr)
    ids, bodies = _bm25_corpus(conn, table, cache, body_expr)
    body_of = dict(zip(ids, bodies))
    ce = _get_cross_encoder()

    def retrieve(question: str, k: int = K) -> list[int]:
        fused = ranked(question, RRF_CE_CANDIDATES)
        if not fused:
            return []
        cands = [(id, body_of[id]) for id, _ in fused]
        return _cross_encoder_rerank(ce, question, cands, k)

    return retrieve
