# Research Log

Automated improvement log maintained by Auto-Researcher.
Each run appends a dated entry describing what was implemented, what was skipped, and why.

---

## 2026-04-22 — Auto-Researcher v4

**Resume score at the start of this run:** 75/100 (top-3: RAG + HyDE + cross-encoder rerank on SEC filings, FastAPI + Streamlit, table-aware KPI extraction, Excel export).

**Implemented (branch `claude/admiring-davinci-QFlkH`):**
- Added `.github/workflows/ci.yml`: installs `requirements.txt`, runs `python -m compileall backend frontend tests`, and executes the documented `python -m unittest discover -s tests -v` suite on pushes and PRs against `main`. Tests in `tests/test_app.py` patch every external dependency (NVIDIA LLM, EDGAR, embedder, reranker, Chroma) via `unittest.mock.patch`, so the workflow runs fully offline with placeholder env vars. Python 3.11 only — matches `requirements.txt` pins.
- Seeded this `RESEARCH_LOG.md` so future auto-researcher runs remember prior work on this repo.

**Why this was prioritized:**
FinLens had no `.github/` directory at all — a RAG + FastAPI project with a 29KB test suite was shipping without a PR gate. Adding CI turns the documented `python -m unittest discover -s tests -v` command into an automatic regression check. Highest-leverage, lowest-risk change for a repo in its first week of existence and with 0 stars — a visible green CI badge materially improves the README's resume signal.

**Evaluated and skipped this run:**
- Multi-Python matrix (3.11 + 3.12): `requirements.txt` pins `chromadb==0.6.3` and `sentence-transformers==5.2.2`, which have not been validated against 3.12 in this repo. Holding at 3.11 until a compatibility pass runs.
- `ruff` lint job: `pyproject.toml` does not yet exist; introducing ruff without a per-file allowlist would flood PRs with style findings. Queued for a future scoped run.
- Dockerfile + docker-compose: README only documents a local-venv flow, and a single-service `uvicorn + streamlit` pair doesn't clearly benefit from compose yet.
- README polish (screenshots, badges): holding until there is at least one green CI run to link a badge against.

**Next-run candidates:**
- Add a CI badge to the top of `README.md` pointing at the new workflow.
- Introduce `ruff` via a `pyproject.toml` with an initial allowlist, then tighten.
- Add a `Dockerfile` (multi-stage: backend + frontend) and a `docker-compose.yaml` so the README `Quick Start` works from a single command.
- Add `make install / test / lint` helper targets so Linux/macOS users get one-liners that mirror the existing PowerShell snippets.
- Add a short nightly workflow that smoke-tests a real EDGAR ingest against a tiny fixture filing (guarded by a manual `workflow_dispatch`).
