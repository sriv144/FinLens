# Research Log

Persistent memory used by the auto-researcher agent. Each run appends a dated
section so future runs can see what was evaluated, what shipped, and what was
deliberately skipped.

## 2026-04-23 - Auto-Researcher v4

**Resume-worthiness score at start of run: 73 / 100**
- Tech stack prestige: 18/25 (FastAPI + Streamlit + ChromaDB + cross-encoder rerank + HyDE - solid RAG stack)
- Commit recency: 23/25 (last push 2026-04-17)
- Feature completeness: 16/20 (ingest + Q&A + YoY compare + metrics + Excel export + Swagger)
- Stars / visibility: 3/15 (0 stars, newest repo of the group)
- README quality: 13/15 (API table, config table, example calls, test + smoke plan)

### Branch
`claude/admiring-davinci-ONkrp`

### Status this run
**Not selected for implementation.** Repo ranked 4th of 6 and the open
claude/admiring-davinci-QFlkH branch already ships the single highest-
leverage addition for this codebase (a GitHub Actions CI workflow that
runs pytest/unittest + compileall against mocked externals). Shipping
a parallel unreviewed branch on top would fragment review surface for
no benefit.

### Prior claude/* branches observed (unmerged on main)
- `claude/admiring-davinci-QFlkH` - CI workflow + RESEARCH_LOG seed
  (commit `4a15e82`). Should be merged first.

### Why this repo scored below the top 3
Strong repo but very new (created 2026-04-17), so visibility and star
count are low. The README is already quite polished and a CI workflow
is already on deck via QFlkH. The remaining gaps (live demo, more
test coverage, additional filing types) all require either deploy
infrastructure or real dataset work - not appropriate for a single
safe-by-default run.

### Evaluated and skipped
- **New CI workflow** - already owned by QFlkH.
- **Dockerfile + compose for local ChromaDB**: useful but requires a
  baseline model-pull story (sentence-transformers weights) to not
  balloon image size.
- **Add an evals harness for retrieval quality**: the right long-term
  move, but needs a gold-standard Q&A dataset on SEC filings that
  this repo does not yet own.
- **Swap NVIDIA-hosted LLM for Anthropic Claude**: the hard constraint
  for this agent is Claude-only models, and this repo currently
  targets NVIDIA-hosted OpenAI-compatible endpoints. Migrating is
  potentially high-value but is a non-trivial code change - queued
  for a dedicated run.

### Next-run candidates
1. Merge QFlkH's CI workflow, then add a `tests/retrieval_eval/` harness
   with 10-20 hand-labeled (question, expected_source) pairs over a
   single filing (e.g. AAPL 10-K) so retrieval regressions surface.
2. Migrate the LLM layer to Anthropic Claude (`claude-sonnet-4-6`)
   behind an env flag so reviewers without NVIDIA credentials can try
   the repo.
3. Add a `docker-compose.yaml` that stands up FastAPI + Streamlit +
   a persistent ChromaDB volume in one command.
4. Record a 60s demo (ingest AAPL 10-K -> ask revenue question ->
   YoY compare with 10-K-1 -> Excel download) and embed at the top of
   README.
5. Publish to a public demo (HF Space, Streamlit Cloud) - visibility
   is this repo's weakest scored signal.
