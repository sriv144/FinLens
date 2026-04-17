"""
Ingestion router — two paths:

  POST /ingest/edgar   →  fetch by ticker + year from SEC EDGAR, parse, embed, store
  POST /ingest/upload  →  accept PDF upload with financial metadata, parse, embed, store
  GET  /ingest/filing-list/{ticker}  →  list available filings for a ticker
"""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.models.schemas import EdgarIngestRequest, IngestResponse
from backend.services import edgar_client, financial_parser, embedder, vector_store

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_NUMERIC_INDICATOR = [
    "revenue", "income", "profit", "loss", "margin", "eps", "earnings",
    "cash", "asset", "debt", "equity", "expense", "capex",
]


def _generate_doc_id() -> str:
    return secrets.token_hex(5)


# ── SEC EDGAR path ─────────────────────────────────────────────────────────────

@router.post("/edgar", response_model=IngestResponse)
def ingest_from_edgar(req: EdgarIngestRequest):
    """
    Fetch a filing from SEC EDGAR by ticker + year, parse it with financial-aware
    chunking (parent-child + table extraction), embed all chunks, and store in
    ChromaDB.
    """
    ticker = req.ticker.upper().strip()
    form_type = req.form_type.upper()

    try:
        if req.year:
            filing_meta, html = edgar_client.fetch_filing_by_year(
                ticker, req.year, form_type
            )
        else:
            filing_meta, html = edgar_client.fetch_latest_10k(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"EDGAR fetch failed: {exc}"
        )

    return _process_and_store(html_or_pdf=html, is_html=True, filing_meta=filing_meta)


@router.get("/filing-list/{ticker}")
def list_available_filings(ticker: str, form_type: str = "10-K", max_filings: int = 5):
    """List recent filings for a ticker without downloading content."""
    try:
        filings = edgar_client.list_filings(
            ticker.upper(), form_type=form_type, max_filings=max_filings
        )
        return {"ticker": ticker.upper(), "filings": filings}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"EDGAR error: {exc}")


# ── PDF upload path ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=IngestResponse)
async def ingest_pdf_upload(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    company_name: str = Form(""),
    fiscal_year: int = Form(...),
    filing_type: str = Form("10-K"),
    filing_date: str = Form(""),
):
    """
    Accept a PDF filing upload, parse it with financial-aware chunking,
    embed all chunks, and store in ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    filing_meta = {
        "ticker": ticker.upper(),
        "company_name": company_name or ticker.upper(),
        "fiscal_year": fiscal_year,
        "form": filing_type,
        "filing_date": filing_date,
    }
    return _process_and_store(html_or_pdf=pdf_bytes, is_html=False, filing_meta=filing_meta)


# ── Shared processing logic ────────────────────────────────────────────────────

def _process_and_store(
    html_or_pdf,
    is_html: bool,
    filing_meta: dict,
) -> IngestResponse:
    """Parse → embed → store a filing and return ingestion stats."""

    # 1. Parse
    if is_html:
        parsed = financial_parser.parse_html_filing(html_or_pdf, filing_meta)
    else:
        parsed = financial_parser.parse_pdf_filing(html_or_pdf, filing_meta)

    if not parsed.children:
        raise HTTPException(
            status_code=422,
            detail="Document parsed but produced no text chunks. "
                   "The filing may be image-only or empty.",
        )

    doc_id = _generate_doc_id()

    # 2. Embed and store children (retrieval index)
    child_texts = [c.text for c in parsed.children]
    child_embeddings = embedder.embed_texts(child_texts)
    vector_store.store_parsed_document(doc_id, parsed.children, child_embeddings, "child")

    # 3. Embed and store parents (context expansion)
    if parsed.parents:
        parent_texts = [p.text for p in parsed.parents]
        parent_embeddings = embedder.embed_texts(parent_texts)
        vector_store.store_parsed_document(doc_id, parsed.parents, parent_embeddings, "parent")

    # 4. Embed and store tables (numeric Q&A)
    if parsed.tables:
        table_texts = [t.text for t in parsed.tables]
        table_embeddings = embedder.embed_texts(table_texts)
        vector_store.store_parsed_document(doc_id, parsed.tables, table_embeddings, "table")

    # 5. Store document-level metadata
    vector_store.store_document_metadata(
        doc_id,
        ticker=filing_meta.get("ticker", ""),
        company_name=filing_meta.get("company_name", ""),
        fiscal_year=filing_meta.get("fiscal_year", 0),
        filing_type=filing_meta.get("form", "10-K"),
        filing_date=filing_meta.get("filing_date", ""),
        parent_count=len(parsed.parents),
        child_count=len(parsed.children),
        table_count=len(parsed.tables),
    )

    return IngestResponse(
        doc_id=doc_id,
        ticker=parsed.ticker,
        company_name=parsed.company_name,
        fiscal_year=parsed.fiscal_year,
        filing_type=parsed.filing_type,
        filing_date=parsed.filing_date,
        parent_count=len(parsed.parents),
        child_count=len(parsed.children),
        table_count=len(parsed.tables),
        message=(
            f"Successfully ingested {parsed.ticker} {parsed.filing_type} "
            f"{parsed.fiscal_year}: {len(parsed.children)} child chunks, "
            f"{len(parsed.tables)} table chunks."
        ),
    )
