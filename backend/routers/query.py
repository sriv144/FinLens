"""
Financial query router with HyDE + cross-encoder reranking.

Retrieval pipeline
──────────────────
1.  HyDE  — LLM generates a hypothetical filing passage for the question.
            The HyDE embedding is used for retrieval instead of the raw question.
            This closes the vocabulary gap between analyst questions and SEC language.

2.  Multi-doc retrieval — child chunks are queried from all requested doc IDs,
            table chunks are always checked for numeric-leaning questions.

3.  Cross-encoder reranking — a ms-marco cross-encoder scores every (question, chunk)
            pair precisely.  Top-K survivors are selected.

4.  Parent expansion — each surviving child's parent_id is looked up, and the
            full parent chunk is returned as context to the LLM.

5.  Generation — NVIDIA LLM generates a grounded, cited answer.

6.  Evaluation — faithfulness and relevance are scored (RAGAS-style).
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from backend.config import TOP_K_CANDIDATES, TOP_K_RESULTS, USE_HYDE
from backend.models.schemas import (
    FinancialQueryRequest,
    FinancialQueryResponse,
    FinancialSource,
    EvaluationScores,
)
from backend.services import embedder, vector_store
from backend.services import llm as llm_service
from backend.services import reranker as reranker_service
from backend.services import evaluator

router = APIRouter(prefix="/query", tags=["query"])

_NUMERIC_WORDS = re.compile(
    r"\b(revenue|income|profit|loss|eps|earnings|margin|cash|debt|asset"
    r"|capex|expense|dividend|buyback|return|ratio|growth|billion|million)\b",
    re.I,
)


def _is_numeric_question(question: str) -> bool:
    return bool(_NUMERIC_WORDS.search(question))


@router.post("", response_model=FinancialQueryResponse)
@router.post("/", response_model=FinancialQueryResponse)
def query_financial(req: FinancialQueryRequest):
    """
    Answer a financial question against one or more ingested filings using
    HyDE + parent-child retrieval + cross-encoder reranking.
    """
    if not req.doc_ids:
        raise HTTPException(status_code=400, detail="At least one doc_id is required.")

    # ── Step 1: HyDE query expansion ──────────────────────────────────────────
    hyde_doc = None
    retrieval_text = req.question
    if req.use_hyde and USE_HYDE:
        try:
            hyde_doc = llm_service.generate_hyde_document(req.question)
            retrieval_text = hyde_doc  # embed the hypothetical passage
        except Exception:
            pass  # graceful fallback to raw question

    query_embedding = embedder.embed_query(retrieval_text)
    raw_query_embedding = embedder.embed_query(req.question)

    # ── Step 2: Multi-doc retrieval ───────────────────────────────────────────
    all_candidates: list[dict] = []
    is_numeric = _is_numeric_question(req.question)

    for doc_id in req.doc_ids:
        # Child chunks (semantic retrieval)
        children = vector_store.query_children(
            doc_id,
            query_embedding,
            top_k=TOP_K_CANDIDATES,
            section_filter=req.section_filter,
        )
        all_candidates.extend(children)

        # Table chunks (injected for numeric questions)
        if is_numeric:
            tables = vector_store.query_tables(doc_id, raw_query_embedding, top_k=5)
            all_candidates.extend(tables)

    if not all_candidates:
        raise HTTPException(
            status_code=404,
            detail="No content found. Make sure the document IDs are valid.",
        )

    # ── Step 3: Cross-encoder reranking ──────────────────────────────────────
    try:
        reranked = reranker_service.rerank(
            query=req.question,
            candidates=all_candidates,
            top_k=req.top_k * 2,  # keep extra for parent expansion
        )
    except Exception:
        # Fallback: sort by distance if reranker unavailable
        reranked = sorted(all_candidates, key=lambda x: x.get("distance", 1.0))[: req.top_k * 2]

    # ── Step 4: Parent chunk expansion ────────────────────────────────────────
    context_chunks: list[str] = []
    source_metas: list[dict] = []

    for cand in reranked[: req.top_k]:
        meta = cand.get("metadata", {})
        parent_id = meta.get("parent_id")
        doc_id = meta.get("doc_id", req.doc_ids[0])

        if parent_id and not cand.get("is_table"):
            parents = vector_store.get_parents_by_ids(doc_id, [parent_id])
            if parents:
                context_chunks.append(parents[0]["text"])
                combined_meta = {**meta, **(parents[0].get("metadata", {}))}
                source_metas.append(combined_meta)
                continue

        # Use child or table text directly
        context_chunks.append(cand["text"])
        source_metas.append(meta)

    # Deduplicate context
    seen_texts: set[str] = set()
    unique_chunks: list[str] = []
    unique_metas: list[dict] = []
    for text, meta in zip(context_chunks, source_metas):
        key = text[:100]
        if key not in seen_texts:
            seen_texts.add(key)
            unique_chunks.append(text)
            unique_metas.append(meta)

    # ── Step 5: LLM generation ────────────────────────────────────────────────
    first_meta = unique_metas[0] if unique_metas else {}
    try:
        answer = llm_service.answer_financial_question(
            context_chunks=unique_chunks,
            question=req.question,
            ticker=first_meta.get("ticker", ""),
            filing_type=first_meta.get("filing_type", ""),
            fiscal_year=first_meta.get("fiscal_year", ""),
            metadatas=unique_metas,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

    # ── Step 6: Evaluation ────────────────────────────────────────────────────
    eval_scores = None
    try:
        scores = evaluator.evaluate_response(req.question, answer, unique_chunks)
        eval_scores = EvaluationScores(**scores)
    except Exception:
        pass

    # ── Build response ────────────────────────────────────────────────────────
    sources = [
        FinancialSource(
            text=cand["text"][:500],
            ticker=cand.get("metadata", {}).get("ticker", ""),
            fiscal_year=cand.get("metadata", {}).get("fiscal_year", 0),
            section_type=cand.get("metadata", {}).get("section_type", ""),
            page_num=cand.get("metadata", {}).get("page_num", 0),
            chunk_kind=cand.get("metadata", {}).get("chunk_kind", ""),
            rerank_score=cand.get("rerank_score"),
            is_table=cand.get("is_table", False),
        )
        for cand in reranked[: req.top_k]
    ]

    return FinancialQueryResponse(
        answer=answer,
        sources=sources,
        hyde_document=hyde_doc,
        evaluation=eval_scores,
        doc_ids_searched=req.doc_ids,
        retrieval_stats={
            "total_candidates": len(all_candidates),
            "after_rerank": len(reranked),
            "context_chunks": len(unique_chunks),
            "table_chunks_included": sum(1 for c in reranked if c.get("is_table")),
            "hyde_used": hyde_doc is not None,
        },
    )
