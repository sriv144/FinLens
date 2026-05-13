# Research Log

This log tracks automated research and improvement runs by the
auto-researcher agent.

---

## 2026-05-13 — Auto-Researcher v4

**Resume score at start of run:** 75 / 100
- Tech stack prestige: 18/25 (RAG + SEC EDGAR + FastAPI + Streamlit + ChromaDB)
- Commit recency: 25/25 (active within the last 48 hours)
- Feature completeness: 15/20
- Stars + visibility: 4/15 (no stars yet)
- README quality: 13/15

**Implemented (branch: `claude/admiring-davinci-HHSYf`):**
- `RESEARCH_LOG.md` (this file) so future runs can avoid duplicate work.

**Why no code changes this run:**
- Token budget was spent on the top 3 (embodied-skill-composer,
  AegisQuant, Autonomous-SRE-Agent), which all had higher resume
  scores and a clear missing-CI gap. FinLens was evaluated but did
  not have a single low-risk / high-impact win that fit in the
  remaining budget.

**Next-run candidates (ranked):**
1. Add `.github/workflows/ci.yml` running pytest + ruff (mirror the
   pattern used in AegisQuant this run).
2. Add a Streamlit screenshot / GIF of a KPI extraction run to the
   README — highest visual return for a recruiter skim.
3. Add an Anthropic Claude option alongside the current LLM path,
   gated on `ANTHROPIC_API_KEY` in `.env.example`.
4. Wire a nightly EDGAR-ingest smoke job behind `workflow_dispatch`.
5. Add a typed schema for the KPI extraction output and a JSON
   Schema export for downstream consumers.
