# Eval results (k=5, 50 golden questions)

| Retriever | Chunker | recall@5 | MRR | MAP |
|---|---|---|---|---|
| ts_rank_cd | fixed | 0.55 | 0.36 | 0.36 |
| bm25 | fixed | 0.83 | 0.68 | 0.68 |
| semantic | fixed | 0.89 | 0.66 | 0.66 |
| rrf | fixed | 0.89 | 0.66 | 0.66 |
| rrf_ce | fixed | 0.91 | 0.83 | 0.83 |
| ts_rank_cd | recursive | 0.52 | 0.29 | 0.29 |
| bm25 | recursive | 0.82 | 0.63 | 0.63 |
| semantic | recursive | 0.86 | 0.57 | 0.57 |
| rrf | recursive | 0.88 | 0.66 | 0.66 |
| rrf_ce | recursive | 0.94 | 0.81 | 0.81 |
| ts_rank_cd | markdown | 0.60 | 0.35 | 0.35 |
| bm25 | markdown | 0.88 | 0.68 | 0.68 |
| semantic | markdown | 0.86 | 0.62 | 0.62 |
| rrf | markdown | 0.92 | 0.68 | 0.68 |
| rrf_ce | markdown | 0.92 | 0.81 | 0.81 |

_Skipped (no snippet match): fixed 3, recursive 0, markdown 0_

## Query variants: rrf_ce base (markdown_context + BM25/semantic RRF + cross-encoder), k=5, golden questions 1-10 (Cloud Run Runtime)

| Retriever | recall@5 | MRR | MAP |
|---|---|---|---|
| rrf_ce (baseline) | 1.00 | 1.00 | 1.00 |
| rewrite | 1.00 | 0.93 | 0.93 |
| multiquery | 1.00 | 1.00 | 1.00 |
| hyde | 1.00 | 1.00 | 1.00 |

_NOT comparable to the 50-question averages above — 10-question subset, contextual table only._
