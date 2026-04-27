# Research Log

This file tracks autonomous research and improvement runs against this
repository.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 71 / 100 — ranked 4 of 6 in the portfolio.

**Branch:** `claude/admiring-davinci-snPHW`.

### Implemented this run

No code changes. This commit only seeds the research log so future runs have
memory of what was already evaluated.

### Why no implementation this run

FinLens is in good shape: clear README with API surface, `.env.example`,
`requirements.txt`, `tests/`, frontend + backend split. The token budget for
this run was prioritized for higher-scoring repos (embodied-skill-composer,
AegisQuant, Autonomous-SRE-Agent), which had clearer high-leverage cleanup
targets (broken root files, missing CI, missing license).

### Evaluated and parked for next run

- **CI workflow** (`pytest tests/`): high-impact, low-risk. The test suite is
  unittest-based and uses FastAPI TestClient — should run cleanly without
  network access if NVIDIA-API calls are mocked. Verify locally first.
- **Dockerfile + docker-compose:** repo is a two-process app (FastAPI +
  Streamlit). A compose file with both services and a shared `.env` would
  shorten onboarding from a multi-terminal flow to `docker compose up`.
- **README screenshot of the Streamlit dashboard:** showing the Q&A and YoY
  comparison views would lift the resume signal substantially.
- **MIT or Apache-2.0 LICENSE:** missing.
- **Replace inline PowerShell-only commands with cross-platform examples:**
  README only shows Windows `Invoke-RestMethod` and PowerShell venv
  activation. Add curl + bash equivalents.

### Next-run candidates (priority order)

1. Add CI (pytest) + LICENSE.
2. Add a docker-compose.yml that brings up backend + frontend + ChromaDB.
3. README screenshot of the Streamlit metrics dashboard.
4. Cross-platform README examples (curl + bash).
