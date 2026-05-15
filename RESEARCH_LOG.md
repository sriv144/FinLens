# Research Log

This file tracks autonomous research and improvement runs against this repo.
Each run lists what was implemented, what was evaluated and skipped, and the
next-run candidate list.

## 2026-05-15 — Auto-Researcher v4

**Resume-worthiness score at start of run: 82 / 100**

Signal breakdown:
- Tech stack prestige: 18/25 (RAG + cross-encoder reranking + HyDE + ChromaDB; finance-domain LLM ops)
- Commit recency: 24/25 (last push 2026-05-11)
- Feature completeness: 18/20 (EDGAR ingest, PDF upload, Q&A, YoY compare, metrics extraction, Excel export)
- Stars + visibility: 8/15
- README quality: 14/15 (full API surface, env reference, smoke-test checklist)

### Implemented this run

Branch: `claude/admiring-davinci-dYRjh`

- `feat(ci)`: added `.github/workflows/ci.yml` that compiles the backend + frontend + tests and runs the existing unittest suite on every push and PR. The README documents the exact test command (`python -m unittest discover -s tests -v`) but never wired it to GH Actions. Uses dummy `NVIDIA_API_KEY` and `EDGAR_USER_AGENT` env vars so external endpoints are not contacted during CI — tests already use `FastAPI TestClient` and do not call out.

### Why this was prioritized

FinLens's selling points are technical depth (HyDE + reranking + parent-child chunking + deterministic KPI extraction). A repo selling depth without CI visibility loses some of that credibility. CI is the lowest-risk improvement available — it does not touch any retrieval path.

### Evaluated and skipped this run

- Swap NVIDIA-hosted OpenAI-compatible chat for Anthropic Claude as the primary LLM — skipped: cross-cutting; the LLM service is reused by Q&A, YoY compare, and metric extraction. Needs a parity benchmark before swapping.
- Add a `ruff` or `black` step — skipped: project does not currently have a configured style baseline; adding one would generate noise diffs unrelated to function.
- Add ChromaDB persistence test using a CI service — skipped: existing tests already mock the vector store contract.

### Next-run candidates

1. Anthropic Claude provider as an alternative LLM behind `LLM_PROVIDER=anthropic`, including a parity benchmark notebook.
2. Add `pytest` + `pytest-cov` alongside unittest (the suite is unittest-style today; pytest can still discover it).
3. Add a screenshot or GIF of the Streamlit metrics dashboard in README — visual artifacts move stars more than text alone.
4. Promote the test step out of `continue-on-error` once the CI environment is verified to match local.
