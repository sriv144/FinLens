# Research Log

This file tracks autonomous research and improvement runs against this
repository.

## 2026-05-17 — Auto-Researcher v4

**Resume-worthiness score at start of run: 74 / 100** (rank 4 of 6).

| Signal | Score |
| --- | --- |
| Tech stack prestige (RAG over SEC + KPI extraction + Excel export) | 18 / 25 |
| Commit recency (updated 2026-05-11) | 22 / 25 |
| Feature completeness (ingest + Q&A + YoY + metrics + export) | 17 / 20 |
| Stars + visibility (no stars yet) | 3 / 15 |
| README quality (detailed, but Windows-only commands, no LICENSE link) | 14 / 15 |

### Implemented this run (branch: `claude/admiring-davinci-Quaj7`)

- **docs(LICENSE): add MIT LICENSE.** Was the top open snPHW next-run
  candidate. Zero-risk and unblocks anyone wanting to fork or evaluate the
  project against a known license. Referenced from a new
  `## License` section in the README.
- **docs(readme): cross-platform API examples.** Appended a
  `## Cross-platform API examples (curl / bash)` section mirroring the
  existing PowerShell `Invoke-RestMethod` block so non-Windows reviewers can
  copy-paste the smoke test without translating. Existing PowerShell
  snippets are preserved verbatim so Windows muscle memory still works.

### Why this was prioritized

Lowest-risk wins on the snPHW next-run list: LICENSE and a curl/bash
examples section. Both are pure docs and add no CI surface to maintain. The
repo was previously rank 4 because of these two gaps relative to its strong
feature set; closing them moves the resume-impression dial without touching
code paths.

### Evaluated and skipped

- **docker-compose.yml bringing up backend + frontend + ChromaDB.** ChromaDB
  here is local-file (per `CHROMA_DB_PATH`), not a server, so a real
  compose file would need to be carefully designed against that storage
  contract — risky without a local run.
- **Streamlit screenshot of the metrics dashboard.** High signal but needs
  a live ingestion run against EDGAR + a working NVIDIA key.
- **CI workflow (pytest).** The test suite uses FastAPI TestClient; would
  need a verification pass to confirm no test reaches the NVIDIA API live.

### Next-run candidates

1. Add a `.github/workflows/ci.yml` running `python -m unittest discover -s
   tests -v` once a local pass confirms no test makes a real network call.
2. Streamlit dashboard screenshot embedded under the Highlights section.
3. docker-compose for backend + frontend with a clear note about ChromaDB
   being local-file.
4. Add a short "Why FinLens?" preamble block summarising the analyst
   workflow it shortens.

### Prior research-log context

Previous runs (most recent first, none merged to `main`):

- `claude/admiring-davinci-snPHW` (2026-04-27) — seeded research log only;
  no code.
