# Research Log

This file tracks autonomous research and improvement runs against this
repository.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 71 / 100 — ranked 4 of 6 in the portfolio.

**Branch:** `claude/admiring-davinci-snPHW`.

### Implemented this run

No code changes. This commit only seeded the research log.

### Next-run candidates (priority order)

1. Add CI (pytest) + LICENSE.
2. Add a docker-compose.yml that brings up backend + frontend + ChromaDB.
3. README screenshot of the Streamlit metrics dashboard.
4. Cross-platform README examples (curl + bash).

## 2026-05-14 — Auto-Researcher v4

**Resume score at start of run:** 71 / 100 — still ranked 4 of 6, but the
top three repos (embodied-skill-composer, AegisQuant, Autonomous-SRE-Agent)
all have prior auto-research branches still awaiting merge to main, so FinLens
is the highest-impact place to land net-new work this run.

**Branch:** `claude/admiring-davinci-5vVWf`.

### Implemented

- **CI workflow** at `.github/workflows/ci.yml`. Installs `requirements.txt`
  on Python 3.11 and 3.12, byte-compiles `backend/`, `frontend/`, and
  `tests/`, then runs `python -m unittest discover -s tests -v`. Sets dummy
  `NVIDIA_API_KEY`, `EDGAR_USER_AGENT`, and per-job temp paths for
  `CHROMA_DB_PATH` / `UPLOAD_DIR` so the FastAPI TestClient suite can boot
  without leaking state into the workspace.
- **`Dockerfile`** (multi-purpose Python 3.11-slim image) with the system
  packages PyMuPDF / lxml / sentence-transformers need at runtime.
- **`docker-compose.yml`** bringing up the FastAPI backend on `:8000` and the
  Streamlit UI on `:8501`, both reading `.env`, sharing a built image, and
  persisting `chroma_db` and `uploads` to named volumes. Onboarding goes from
  the multi-terminal PowerShell flow in the README to `docker compose up`.
  Includes a `/health` healthcheck on the backend.
- **`.dockerignore`** so build context skips `.venv`, caches, `chroma_db`,
  and uploads.
- **MIT `LICENSE`.**

### Why this was prioritized

This is the exact priority list parked from 2026-04-27: CI + LICENSE +
docker-compose. CI is the single highest-leverage signal because FinLens
already has a real unittest suite using FastAPI TestClient — the prior log
verified it should run cleanly without network if NVIDIA calls are mocked.
The docker-compose adds the second-largest UX delta: two-process apps without
compose are a notorious onboarding wall.

### Evaluated and skipped

- **README screenshot of the Streamlit metrics dashboard:** still parked
  because this autonomous run cannot capture screenshots. Will require a
  human-supplied PNG checked into `docs/`.
- **Cross-platform README examples (curl + bash):** worth doing, but a
  README rewrite alongside the docker-compose docs is a separate, larger PR;
  the current PowerShell-only quickstart still works on Windows.
- **Rate limiting on `/query` and `/ingest/edgar`:** good idea, but adding a
  middleware now without load profiling risks under- or over-tuning the
  limits. Deferred.

### Next-run candidates

1. README screenshot under a new `docs/screenshots/` folder + reference it
   from the README.
2. Add `bash`/`curl` equivalents next to the existing PowerShell examples.
3. Add `slowapi` rate limiting to `/query` and `/ingest/edgar` after
   measuring real call rates.
4. Optional: pin Streamlit + ChromaDB versions in `Dockerfile` ahead of
   `requirements.txt` so cached layers survive minor bumps.
