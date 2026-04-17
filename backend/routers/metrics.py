"""
Financial metrics extraction router.

  POST /metrics/extract/{doc_id}   ->  refresh key KPIs from an ingested filing
  GET  /metrics/{doc_id}           ->  return cached metrics, extracting if absent
"""

from __future__ import annotations

import json
import os
import re

from fastapi import APIRouter, HTTPException

from backend.config import CHROMA_DB_PATH
from backend.models.schemas import FinancialMetrics
from backend.services import vector_store
from backend.services import llm as llm_service

router = APIRouter(prefix="/metrics", tags=["metrics"])

_CACHE_DIR = os.path.join(CHROMA_DB_PATH, "_metrics_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(doc_id: str) -> str:
    return os.path.join(_CACHE_DIR, f"{doc_id}.json")


def _read_cache(doc_id: str) -> dict | None:
    path = _cache_path(doc_id)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _write_cache(doc_id: str, data: dict) -> None:
    try:
        with open(_cache_path(doc_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _empty_metrics() -> dict:
    return {
        "revenue": None,
        "revenue_unit": None,
        "net_income": None,
        "net_income_unit": None,
        "eps_diluted": None,
        "gross_margin_pct": None,
        "operating_margin_pct": None,
        "total_assets": None,
        "total_assets_unit": None,
        "cash_and_equivalents": None,
        "cash_unit": None,
        "long_term_debt": None,
        "debt_unit": None,
        "r_and_d_expense": None,
        "r_and_d_unit": None,
        "employees": None,
        "key_risks": [],
        "strategic_highlights": [],
    }


def _parse_number(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("$", "").strip()
    if cleaned in {"", "-", "--"}:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    try:
        parsed = float(cleaned)
        return -parsed if negative else parsed
    except ValueError:
        return None


def _numbers_after(label: str, text: str) -> list[float]:
    pattern = re.compile(
        rf"{label}\s*(?:\|\s*)+(?:\$?\s*\|\s*)?(\(?\$?[\d,]+(?:\.\d+)?\)?)",
        re.I,
    )
    values: list[float] = []
    for match in pattern.finditer(text):
        value = _parse_number(match.group(1))
        if value is not None:
            values.append(value)
    return values


def _first_number_after(label: str, text: str) -> float | None:
    values = _numbers_after(label, text)
    return values[0] if values else None


def _first_large_number_after(label: str, text: str) -> float | None:
    for value in _numbers_after(label, text):
        if abs(value) >= 100:
            return value
    return None


def _largest_large_number_after(label: str, text: str) -> float | None:
    values = [value for value in _numbers_after(label, text) if abs(value) >= 100]
    return max(values, key=abs) if values else None


def _first_percent_after(label: str, text: str) -> float | None:
    for value in _numbers_after(label, text):
        if 0 <= value <= 100:
            return value
    return None


def _extract_table_metrics(text: str) -> dict:
    metrics = _empty_metrics()
    unit = (
        "USD millions"
        if re.search(r"\(\s*(?:\$?\s*)?in millions", text, re.I) or "$" in text
        else None
    )

    revenue = (
        _first_large_number_after(r"\bTotal net sales\b", text)
        or _first_large_number_after(r"\bTotal revenues?\b", text)
        or _first_large_number_after(r"(?<!Deferred )\bRevenue\b(?!\s+by)", text)
    )
    if revenue is not None:
        metrics["revenue"] = revenue
        if unit:
            metrics["revenue_unit"] = unit

    gross_margin_amount = _first_large_number_after(r"\bGross margin\b", text)
    operating_income_amount = _first_large_number_after(r"\bOperating income\b", text)

    row_specs = [
        ("net_income", "net_income_unit", r"\bNet income\b"),
        ("total_assets", "total_assets_unit", r"\bTotal assets\b"),
        ("cash_and_equivalents", "cash_unit", r"\bCash and cash equivalents\b"),
        ("long_term_debt", "debt_unit", r"\bLong[- ]term debt\b"),
    ]
    for field, unit_field, label in row_specs:
        value = _first_large_number_after(label, text)
        if value is not None:
            metrics[field] = value
            if unit:
                metrics[unit_field] = unit

    if metrics["long_term_debt"] is None:
        term_debt = _largest_large_number_after(r"\bTerm debt\b", text)
        if term_debt is not None and abs(term_debt) >= 1000:
            metrics["long_term_debt"] = term_debt
            if unit:
                metrics["debt_unit"] = unit

    r_and_d = _largest_large_number_after(r"(?<!Capitalized )\bResearch and development\b", text)
    if r_and_d is not None:
        metrics["r_and_d_expense"] = r_and_d
        if unit:
            metrics["r_and_d_unit"] = unit

    eps = _first_number_after(r"\b(?:Net income per diluted share|Diluted)\b", text)
    if eps is not None and eps < 1000:
        metrics["eps_diluted"] = eps

    gross_margin = _first_percent_after(r"\bGross margin\b", text)
    if gross_margin is not None and gross_margin <= 100:
        metrics["gross_margin_pct"] = gross_margin
    elif revenue and gross_margin_amount and gross_margin_amount > 100:
        metrics["gross_margin_pct"] = round((gross_margin_amount / revenue) * 100, 1)

    operating_margin = _first_percent_after(r"\bOperating income\b", text)
    if operating_margin is not None and operating_margin <= 100:
        metrics["operating_margin_pct"] = operating_margin
    elif revenue and operating_income_amount and operating_income_amount > 100:
        metrics["operating_margin_pct"] = round((operating_income_amount / revenue) * 100, 1)

    return metrics

def _merge_metrics(primary: dict, fallback: dict) -> dict:
    merged = {**fallback, **(primary or {})}
    for key, value in fallback.items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def _build_metrics_context(doc_id: str, query_embedding: list[float]) -> str:
    tables = vector_store.query_tables(doc_id, query_embedding, top_k=20)
    children = vector_store.query_children(doc_id, query_embedding, top_k=20)
    return "\n\n".join(c["text"] for c in (tables + children))


@router.post("/extract/{doc_id}", response_model=FinancialMetrics)
def extract_metrics(doc_id: str):
    """
    Refresh structured financial KPIs from an already-ingested document.
    """
    doc_meta = vector_store.get_document_metadata(doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    from backend.services import embedder

    metric_query = (
        "annual revenue net income earnings per diluted share gross margin "
        "operating income total assets cash and cash equivalents long-term debt "
        "research and development expense consolidated statements of income balance sheets"
    )
    q_emb = embedder.embed_query(metric_query)
    all_text = _build_metrics_context(doc_id, q_emb)
    table_metrics = _extract_table_metrics(all_text)

    try:
        raw_metrics = llm_service.extract_financial_metrics(all_text)
    except Exception as exc:
        if any(v not in (None, "", []) for v in table_metrics.values()):
            raw_metrics = {}
        else:
            raise HTTPException(
                status_code=502,
                detail=f"LLM metrics extraction failed: {exc}",
            ) from exc

    merged_metrics = _merge_metrics(raw_metrics, table_metrics)
    result = FinancialMetrics(
        doc_id=doc_id,
        ticker=doc_meta.get("ticker", ""),
        fiscal_year=doc_meta.get("fiscal_year", 0),
        filing_type=doc_meta.get("filing_type", "10-K"),
        **{
            k: v for k, v in merged_metrics.items()
            if k in FinancialMetrics.model_fields and k not in {"doc_id", "ticker", "fiscal_year", "filing_type"}
        },
    )

    _write_cache(doc_id, result.model_dump())
    return result


@router.get("/{doc_id}", response_model=FinancialMetrics)
def get_metrics(doc_id: str):
    """Return cached metrics for a document. Extract first if not cached."""
    cached = _read_cache(doc_id)
    if cached:
        return FinancialMetrics(**cached)
    return extract_metrics(doc_id)





