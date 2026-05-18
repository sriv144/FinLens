# Research Log

A running log of autonomous research-and-development cycles on FinLens.

---

## 2026-05-18 — Auto-Researcher v4

**Resume score at start of run:** 47/100
**Branch:** `claude/admiring-davinci-9dHiZ`
**Status:** Evaluated, no code change shipped this cycle.

### Why skipped this cycle
- `main` has only two commits (initial release + encoding polish, dated 2026-04-17).
  No real recent activity to diff against.
- 0 stars, 0 forks — the lowest visibility of the 6 target repos this cycle, so the
  marginal resume return on a polish pass is lower than for AegisQuant or the SRE agent.
- Budget went to the top three (AegisQuant polish, SRE → Claude migration,
  embodied-skill-composer CI + LICENSE + CITATION).

### Candidates evaluated
- **Migrate the RAG reasoning model to Claude** — high value; deferred until the
  code path is fully read so the migration can stay surgical (see SRE-Agent precedent).
- **Add screenshots of the Streamlit UI to README** — needs the app running; deferred.
- **CI workflow (pytest + ruff)** — a safe single ship; deferred only because budget
  went to the top three.

### Next-run candidates
1. Read `src/`, confirm the RAG backend, then ship a Claude-Sonnet-4.6 reasoning
   migration mirroring the Autonomous-SRE-Agent pattern (this cycle).
2. Add `LICENSE` (MIT), `CITATION.cff`, and a CI workflow that exercises the
   document-chunk + KPI-extract pipeline on a small sample 10-K.
3. Add a screenshots/GIF section to README.
4. Add a CLI entry point (`finlens analyze <filing.html>`) for non-UI users.
5. Add Excel-export regression tests.
