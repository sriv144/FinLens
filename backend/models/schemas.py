"""
Pydantic schemas for the current FinLens API.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EdgarIngestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'")
    year: Optional[int] = Field(None, description="Fiscal year to ingest. Omit for latest.")
    form_type: str = Field("10-K", description="SEC filing type, e.g. '10-K' or '10-Q'")


class IngestResponse(BaseModel):
    doc_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    filing_type: str
    filing_date: str
    parent_count: int
    child_count: int
    table_count: int
    message: str


class FinancialQueryRequest(BaseModel):
    question: str
    doc_ids: list[str] = Field(..., description="One or more ingested document IDs to search")
    section_filter: Optional[str] = Field(
        None,
        description=(
            "Optional section restriction: business, risk_factors, mda, "
            "financial_statements, or notes."
        ),
    )
    use_hyde: bool = Field(True, description="Use HyDE query expansion before retrieval")
    top_k: int = Field(5, ge=1, le=20)


class FinancialSource(BaseModel):
    text: str
    ticker: str = ""
    fiscal_year: int = 0
    section_type: str = ""
    page_num: int = 0
    chunk_kind: str = ""
    rerank_score: Optional[float] = None
    is_table: bool = False


class EvaluationScores(BaseModel):
    faithfulness: float = Field(..., description="0-1 score for grounding quality")
    relevance: float = Field(..., description="0-1 score for answer relevance")
    quality_score: float = Field(..., description="0-1 composite quality score")


class FinancialQueryResponse(BaseModel):
    answer: str
    sources: list[FinancialSource]
    hyde_document: Optional[str] = Field(None, description="Hypothetical answer used for HyDE retrieval")
    evaluation: Optional[EvaluationScores] = None
    doc_ids_searched: list[str]
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)


class FinancialMetrics(BaseModel):
    doc_id: str
    ticker: str
    fiscal_year: int
    filing_type: str
    revenue: Optional[float] = None
    revenue_unit: Optional[str] = None
    net_income: Optional[float] = None
    net_income_unit: Optional[str] = None
    eps_diluted: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    total_assets: Optional[float] = None
    total_assets_unit: Optional[str] = None
    cash_and_equivalents: Optional[float] = None
    cash_unit: Optional[str] = None
    long_term_debt: Optional[float] = None
    debt_unit: Optional[str] = None
    r_and_d_expense: Optional[float] = None
    r_and_d_unit: Optional[str] = None
    employees: Optional[float] = None
    key_risks: list[str] = Field(default_factory=list)
    strategic_highlights: list[str] = Field(default_factory=list)


class YoYCompareRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    aspect: Optional[str] = Field(
        None,
        description="Optional focus area, e.g. 'risk factors' or 'AI strategy'",
    )


class YoYCompareResponse(BaseModel):
    ticker: str
    year_1: int
    year_2: int
    financial_trends: str
    risk_evolution: str
    strategic_shifts: str
    management_tone: str
    summary: str


class DocumentInfo(BaseModel):
    doc_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    filing_type: str
    filing_date: str
    child_count: int = 0
    table_count: int = 0
