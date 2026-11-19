# Report: rag.py — tracing-free streaming RAG

Plan: `.pi/plans/rag-streaming.md` | Status: COMPLETE

## Summary

Created `rag.py`: tracing-free RAG module — `retrieve`/`rerank` AST-identical to traces.py minus `@observe`/`langfuse.*`, `_sources_block` citation metadata helper, `stream_answer` async generator over `ollama.AsyncClient` yielding content deltas. No langfuse import (direct or transitive), no new deps, zero edits to existing files.

## Tasks

- CREATE `rag.py` (repo root; imports: pipeline/ragas/retrievers/setup + `collections.abc.AsyncIterator`; `CITED_ANSWER_PROMPT` verbatim copy w/ sync comment)

## Tests

No pytest (convention: inline `uv run`). Validation ladder below, all green.

## Validation

- L1 `import rag` → ok
- L2 purity: no `langfuse` in `sys.modules` after import; no `langfuse`/`observe` attrs → `tracing-free: ok`
- L3 parity: `CITED_ANSWER_PROMPT`/`NUM_PREDICT_CITED`/`MODEL` equal traces'; `retrieve`/`rerank` AST-equal to traces minus langfuse call statements; live golden-query equivalence (`hits == tr(q)`, `top == rr(...)`) → `[140, 139, 154, 141, 156]`
- L4 e2e: 74 deltas streamed, `'[1]' in text`, `'Citations:' in text`, citation = cloud_run.md line 983 (matches file: "The SIGTERM signal is a warning…") → green
- L5 no-regression: `main.py --help` ok; `import traces, ragas, main, rag` ok; git shows zero modifications to existing files

## Deviations

- Plan's L3 line-filter source comparison (`norm(a) == norm(b)` on raw lines) failed on two counts: traces' `langfuse.update_current_span(...)` spans multiple lines (continuation lines survive a `'langfuse' in line` filter) and black reformatted traces' signatures. Replaced with AST comparison (parse both, strip decorator lists, drop `langfuse.*` call statements, `ast.dump` equality) — same intent (equivalence modulo tracing), stronger tool.
- Live equivalence run needed `rag._conn.close()` between the two modules: rag's psycopg connection stays idle-in-transaction after its SELECTs (no autocommit), and traces' `ensure_table` `ALTER TABLE` then blocks forever on the held `AccessShareLock`. Test-harness artifact (two connections in one process), not a rag.py defect — documented as a known interaction in the report; no source change.

## Issues

- None blocking. Open (user call, unchanged from plan): CLI streaming command (`asyncio.run` Typer wrapper) — rag.py ships library-only; surfaced token usage from final streamed part — deltas only.