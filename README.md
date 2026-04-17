# FinLens

FinLens is a financial document intelligence app for SEC filings. It ingests annual filings from SEC EDGAR or uploaded PDFs, builds a retrieval index, answers analyst-style questions, compares filings year over year, extracts structured KPIs, and exports metrics to Excel.

## Highlights

- SEC EDGAR ingestion by ticker, filing type, and fiscal year.
- PDF upload fallback for local filing analysis.
- Parent-child chunking for precise retrieval with richer context expansion.
- HyDE query expansion for stronger financial Q&A retrieval.
- Cross-encoder reranking for higher quality source selection.
- Table-aware retrieval and deterministic KPI extraction for financial statements.
- Year-over-year filing comparison across revenue trends, risks, strategy, capital allocation, and management tone.
- Streamlit UI for ingestion, Q&A, comparison, metrics dashboard, and Excel export.
- FastAPI backend with interactive Swagger docs.

## Tech Stack

- Backend: FastAPI, Pydantic, Uvicorn
- Frontend: Streamlit
- Retrieval: ChromaDB, sentence-transformers, cross-encoder reranking
- Filing parsing: SEC EDGAR JSON/archive APIs, BeautifulSoup, lxml, PyMuPDF
- LLM provider: NVIDIA-hosted OpenAI-compatible chat completions
- Reports: openpyxl Excel export
- Tests: unittest with FastAPI TestClient

## Project Structure

```text
backend/
  main.py                 FastAPI app and route registration
  config.py               Environment-backed runtime configuration
  models/schemas.py       Request and response schemas
  routers/                API routes for ingest, query, compare, metrics, export
  services/               EDGAR, parsing, embedding, reranking, LLM, vector store
frontend/
  app.py                  Streamlit application
tests/
  test_app.py             API, parser, vector-store, metrics, and export tests
requirements.txt          Python dependencies
.env.example              Environment template
```

## Requirements

- Python 3.11 or newer recommended.
- A valid NVIDIA API key with access to the configured chat model.
- A SEC-compliant EDGAR user agent in the format `Your Name email@example.com`.

## Quick Start

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/sriv144/FinLens.git
cd FinLens
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least these values:

```dotenv
NVIDIA_API_KEY=nvapi-your-real-key
EDGAR_USER_AGENT=Your Name your.email@example.com
```

## Run Locally

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Open these URLs:

- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

If port `8000` is already in use, run the backend on another port and point Streamlit at it:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
$env:BACKEND_URL="http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8502
```

## API Surface

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health and feature list |
| `GET` | `/documents` | List ingested filings |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/ingest/edgar` | Fetch, parse, embed, and store a filing from SEC EDGAR |
| `GET` | `/ingest/filing-list/{ticker}` | Preview available SEC filings for a ticker |
| `POST` | `/ingest/upload` | Upload and ingest a local PDF filing |
| `POST` | `/query` or `/query/` | Ask questions across one or more ingested filings |
| `POST` | `/compare/yoy` | Compare two filings year over year |
| `POST` | `/metrics/extract/{doc_id}` | Refresh structured KPI extraction for a filing |
| `GET` | `/metrics/{doc_id}` | Read cached metrics, extracting if missing |
| `GET` | `/export/{doc_id}/excel` | Download an Excel metrics report |

## Example API Calls

Ingest the latest Apple 10-K from EDGAR:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ingest/edgar `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","year":0,"form_type":"10-K"}'
```

Ask a question after ingestion:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query/ `
  -ContentType "application/json" `
  -Body '{"question":"What were the main revenue drivers?","doc_ids":["DOC_ID_HERE"],"use_hyde":true,"top_k":5}'
```

Extract metrics:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/metrics/extract/DOC_ID_HERE
```

Download Excel report:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/export/DOC_ID_HERE/excel `
  -UseBasicParsing `
  -OutFile FinLens_metrics.xlsx
```

## Test and Verify

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run a syntax/import compile pass:

```powershell
.\.venv\Scripts\python.exe -m compileall backend frontend tests
```

Manual smoke test:

1. Start backend and frontend.
2. Ingest a 10-K from EDGAR, for example AAPL or NVDA.
3. Ask one qualitative question and one numeric question.
4. Ingest a second filing for the same ticker and run a YoY comparison.
5. Open Metrics Dashboard and click `Extract / Refresh Metrics`.
6. Download the Excel report and confirm it opens.
7. Check `/health`, `/documents`, and `/docs`.

## Configuration

Most settings are controlled with environment variables. Common values:

| Variable | Required | Description |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes | NVIDIA API key for chat completions |
| `NVIDIA_BASE_URL` | No | OpenAI-compatible NVIDIA endpoint |
| `NVIDIA_LLM_MODEL` | No | Chat model name |
| `EDGAR_USER_AGENT` | Yes | SEC-compliant user agent string |
| `CHROMA_DB_PATH` | No | Local ChromaDB path |
| `BACKEND_URL` | No | Frontend target backend URL |
| `UPLOAD_TIMEOUT_SECONDS` | No | Long request timeout for ingestion and analysis |

## Notes

- `.env`, `.venv`, local uploads, and ChromaDB data are intentionally ignored.
- EDGAR may rate-limit or reject requests without a real user-agent name and email.
- Query and comparison features require an NVIDIA key authorized for the configured model.
- Metrics extraction includes deterministic SEC table parsing so KPI cards can still populate when LLM extraction is unavailable.
