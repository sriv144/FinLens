# Research Log

## 2026-04-29 — Auto-Researcher v4

**Resume score at start of run:** 75 / 100. Solid mid-pack: strong RAG
stack (HyDE + cross-encoder rerank + parent-child chunking + table-aware
retrieval), an excellent README, working FastAPI + Streamlit, EDGAR ingestion,
and Excel export.

**Branch:** `claude/admiring-davinci-dI8Uk`

### What was implemented
No code or config changes this run. Seeded this `RESEARCH_LOG.md` so future
autonomous runs have memory of what was evaluated.

### Why this was the deferred outcome
The top-3 cohort (embodied-skill-composer 85, AegisQuant 82,
Autonomous-SRE-Agent 80) and the lowest-readme repo (ai_interview_coach 53,
showcase-enhancement target) consumed this run's token budget. FinLens did
not have an obvious low-risk, high-impact gap once the README and
`.env.example` were already in place. Adding new RAG features (e.g. a second
reranker, a HyDE ablation harness) is high-impact but high-risk without
first running the existing tests in CI to establish a baseline.

### Evaluated and skipped
- **CI workflow.** Worth doing; the `tests/test_app.py` suite is small and
  uses `unittest` + `TestClient`. Skipped this run only because a CI
  workflow that imports `sentence-transformers`, `chromadb`, and
  `cross-encoder` models on every push needs careful caching to avoid
  multi-minute installs. Tagged for next run.
- **Adding a CI badge to the README.** Blocked on the CI workflow.
- **Switching the LLM provider from NVIDIA-NIM to Anthropic Claude.** Out
  of scope for a docs-pass run; would touch `services/llm.py` plus prompt
  templates plus tests.
- **Adding response caching for repeated `/query` calls.** Genuine
  performance win, but needs a cache-key design that respects HyDE expansion
  and reranker outputs.

### Next-run candidates
- Add `.github/workflows/ci.yml` with sentence-transformers model caching.
- Add `LICENSE` (MIT) so the repo is unambiguously open-source.
- Add per-`/query` LRU cache keyed on `(question, doc_ids, use_hyde, top_k)`.
- Optional: provider-pluggable LLM layer (NVIDIA NIM today, Anthropic
  Claude as a second option).
