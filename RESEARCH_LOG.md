# Research Log

A running log of automated improvement runs against this repo.

## 2026-04-28 — Auto-Researcher v4

**Resume score (start of run):** 77 / 100

- Tech stack prestige: 20 (FastAPI + Streamlit, RAG with parent-child chunking, HyDE, cross-encoder rerank, table-aware retrieval)
- Commit recency: 20 (pushed 2026-04-17)
- Feature completeness: 17 (ingest, query, YoY compare, KPI extraction, Excel export)
- Stars / visibility: 6
- README quality: 14 (already strong: API surface, configuration, manual smoke checklist)

### Implemented on `claude/admiring-davinci-jnSqo`

1. **`.github/workflows/tests.yml`** — a CI workflow that:
   - Runs on push and PR to `main`, plus manual dispatch.
   - Installs `requirements.txt`.
   - Runs `python -m compileall backend frontend tests` to catch syntax / import errors across the whole project (matching the manual command the README already documents).
   - Runs `python -m unittest discover -s tests -v` (the repo's chosen test runner per the README).
   - Provides safe dummy values for `NVIDIA_API_KEY`, `EDGAR_USER_AGENT`, and `ANTHROPIC_API_KEY` so import-time env reads don't fail.
   - Uses a concurrency group so superseded pushes auto-cancel.
2. **This `RESEARCH_LOG.md`**.

### Why these were prioritized

- The README already calls out the exact CI-friendly commands (`compileall` + `unittest discover`). Wiring those into GitHub Actions is the smallest, lowest-risk change that produces real recruiter signal (a green test badge).
- Pure additive change — no existing code path is modified, no env keys changed.
- The repo already has `.env.example`, a clean `requirements.txt`, and a structured `tests/` directory, so CI lands cleanly on first attempt.

### Evaluated and skipped this run

- **Adding a Claude / Anthropic provider option** alongside the existing NVIDIA-hosted OpenAI-compatible client. High resume value, but touches `backend/services/` and the LLM-dispatch logic and would benefit from local verification before shipping.
- **Dockerfile / docker-compose**. Useful, but lower-impact than CI for this repo. Logged.
- **README badges** (CI status, Python version). Worth doing after the workflow has produced a stable run. Logged.
- **Performance work on retrieval** (caching the cross-encoder, batching queries). Real optimisation potential, but requires profiling. Out of scope for an unattended run.

### Next-run candidates

- Add CI status badge + Python-version badge to README once `tests.yml` has produced at least one run.
- Add a Claude/Anthropic provider option behind an env flag (keep NVIDIA NIM as default to preserve current behaviour).
- Ship a `Dockerfile` and a `docker-compose.yaml` running backend + frontend + chroma volume.
- Add a `MODEL_CARD.md` documenting which retrieval pieces (HyDE, cross-encoder, parent-child chunking) are on by default and the measured uplift from each.
- Pre-commit config (ruff + mypy) to keep the codebase clean in CI.
