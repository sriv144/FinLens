"""
FinLens - Financial Document Intelligence API

Endpoints:
  POST /ingest/edgar                 Fetch and ingest a filing from SEC EDGAR
  POST /ingest/upload                Upload a PDF filing with metadata
  GET  /ingest/filing-list/{ticker}  List available filings for a ticker
  POST /query                        Financial Q&A with HyDE and reranking
  POST /query/                       Financial Q&A alias
  POST /compare/yoy                  Year-over-year comparison of two filings
  POST /metrics/extract/{id}         Extract structured KPIs from a filing
  GET  /metrics/{id}                 Retrieve cached metrics
  GET  /export/{id}/excel            Download Excel report
  GET  /documents                    List ingested documents
  GET  /health                       Health check
  GET  /docs                         Interactive Swagger UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import compare, export, ingest, metrics, query

app = FastAPI(
    title="FinLens - Financial Document Intelligence",
    description=(
        "Production-grade RAG for SEC filings with HyDE query expansion, "
        "cross-encoder reranking, parent-child chunking, table-aware extraction, "
        "and RAGAS-style evaluation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8502",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(compare.router)
app.include_router(metrics.router)
app.include_router(export.router)


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "service": "FinLens",
        "version": "2.0.0",
        "features": [
            "SEC EDGAR integration",
            "HyDE query expansion",
            "cross-encoder reranking",
            "parent-child chunking",
            "table-aware extraction",
            "RAGAS-style evaluation",
        ],
    }


@app.get("/documents", tags=["documents"])
def list_documents():
    from backend.services.vector_store import list_documents as _list

    return {"documents": _list()}
