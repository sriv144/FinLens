"""
Year-over-year financial comparison router.

  POST /compare/yoy  →  compare two ingested filings (same ticker, different years)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.schemas import YoYCompareRequest, YoYCompareResponse
from backend.services import embedder, vector_store
from backend.services import llm as llm_service

router = APIRouter(prefix="/compare", tags=["comparison"])


@router.post("/yoy", response_model=YoYCompareResponse)
def compare_yoy(req: YoYCompareRequest):
    """
    Compare two filings (typically same company, different years) and return
    a structured year-over-year analysis.
    """
    meta1 = vector_store.get_document_metadata(req.doc_id_1)
    meta2 = vector_store.get_document_metadata(req.doc_id_2)

    if not meta1:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id_1}' not found.")
    if not meta2:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id_2}' not found.")

    ticker = meta1.get("ticker", meta2.get("ticker", "UNKNOWN"))

    # Retrieve relevant passages from both filings
    query_text = req.aspect or (
        "revenue earnings risk factors strategic outlook management discussion"
    )
    q_emb = embedder.embed_query(query_text)

    chunks1 = vector_store.query_children(req.doc_id_1, q_emb, top_k=6)
    chunks2 = vector_store.query_children(req.doc_id_2, q_emb, top_k=6)

    texts1 = [c["text"] for c in chunks1]
    texts2 = [c["text"] for c in chunks2]

    if not texts1 or not texts2:
        raise HTTPException(
            status_code=422,
            detail="One or both documents returned no chunks. Re-ingest and try again.",
        )

    try:
        result = llm_service.compare_filings_yoy(
            chunks1=texts1,
            chunks2=texts2,
            ticker=ticker,
            year1=meta1.get("fiscal_year", 0),
            year2=meta2.get("fiscal_year", 0),
            type1=meta1.get("filing_type", "10-K"),
            type2=meta2.get("filing_type", "10-K"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM comparison failed: {exc}",
        ) from exc

    return YoYCompareResponse(
        ticker=ticker,
        year_1=meta1.get("fiscal_year", 0),
        year_2=meta2.get("fiscal_year", 0),
        **result,
    )
