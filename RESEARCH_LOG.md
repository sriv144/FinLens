# Research Log

A running log of automated research-and-development passes against this repository.

## 2026-05-16 — Auto-Researcher v4

**Resume-worthiness score at start of run: 73 / 100**

| Signal | Score |
| --- | --- |
| Tech stack prestige (RAG + cross-encoder rerank + HyDE + ChromaDB + FastAPI + Streamlit) | 20 / 25 |
| Commit recency (updated 2026-05-11) | 22 / 25 |
| Feature completeness (EDGAR ingest, YoY compare, KPI extract, Excel export) | 18 / 20 |
| Stars + visibility (0 stars) | 0 / 15 |
| README quality (very good — API table, env table, examples) | 13 / 15 |

### Implemented this run

Nothing landed on `claude/admiring-davinci-Ow84F` this pass. FinLens already ships `.env.example`, a tests directory, and a thorough README; the remaining high-impact gaps need more than a safe single-commit autonomous pass can deliver.

### Why no work was done

The repository is currently the most "boring-correct" of the 6 targets: feature-complete, README has an API surface table, env table, example calls, and an explicit test-and-verify section. The honest growth axes from here are not safe single-commit changes:

- **Anthropic / Claude as the LLM provider.** README hardcodes NVIDIA NIM (`NVIDIA_API_KEY`, `NVIDIA_BASE_URL`). Adding Claude as an alternative requires touching `backend/services/` LLM code and re-running the test suite.
- **A GitHub Actions CI workflow.** Promising and safe, but the project pulls heavy ML deps (`sentence-transformers`, cross-encoder) and CI tuning matters — deferred to its own pass so it can be tested.
- **Live demo screenshot of the Streamlit dashboard.** Highest resume value but needs the app actually running.

### Next-run candidates

1. Add a GitHub Actions CI workflow that runs `python -m unittest discover -s tests` plus `compileall backend frontend tests`.
2. Add an Anthropic-backed LLM path (`claude-opus-4-7` / `claude-sonnet-4-6`) selectable via env var, alongside the existing NVIDIA NIM flow.
3. Commit a Streamlit dashboard screenshot to `docs/` and embed it at the top of the README so the visual product is visible on GitHub.
4. Add a Dockerfile + docker-compose stack so the full backend+frontend boots with one command.
5. Strengthen `.gitignore` to keep ChromaDB persistence and uploaded PDFs out of git.
