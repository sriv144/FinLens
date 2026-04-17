"""
LLM service with financial-domain prompts, HyDE query expansion,
structured metrics extraction, and year-over-year comparison.
"""

from __future__ import annotations

import json
import re

import requests

from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_LLM_MODEL


# ── Prompt templates ──────────────────────────────────────────────────────────

FINANCIAL_QA_PROMPT = """You are a senior financial analyst AI.
Answer the QUESTION using ONLY the provided context from the SEC filing.
Do not use outside knowledge.
If the context does not support the answer, say: "This information is not available in the provided filing."

Rules:
- Be precise. Cite specific numbers, dates, and sections where available.
- If you mention a figure, state the unit (millions, billions, %).
- Keep the answer under 300 words unless a detailed breakdown is requested.

Context (from {ticker} {filing_type} {fiscal_year}):
{context}

Question: {question}

Answer:"""

HYDE_GENERATION_PROMPT = """You are a financial analyst with deep expertise in SEC filings.

Generate a SHORT hypothetical excerpt (3-5 sentences) that would PERFECTLY answer
the following question if it appeared in a 10-K or 10-Q filing.
Write in the style of SEC disclosure language.
Do not wrap in quotes. Output the excerpt directly.

Question: {question}

Hypothetical SEC filing excerpt:"""

METRICS_EXTRACTION_PROMPT = """Extract key financial metrics from the following text.
Return a valid JSON object with the structure below.
Use null for any metric not found in the text.
Numbers should be plain floats (e.g. 394.3 for $394.3B). Do NOT include units in the values.

Expected JSON structure:
{{
  "revenue": null,
  "revenue_unit": null,
  "net_income": null,
  "net_income_unit": null,
  "eps_diluted": null,
  "gross_margin_pct": null,
  "operating_margin_pct": null,
  "total_assets": null,
  "total_assets_unit": null,
  "cash_and_equivalents": null,
  "cash_unit": null,
  "long_term_debt": null,
  "debt_unit": null,
  "r_and_d_expense": null,
  "r_and_d_unit": null,
  "employees": null,
  "fiscal_year": null,
  "key_risks": [],
  "strategic_highlights": []
}}

Text:
{text}

JSON:"""

YOY_COMPARISON_PROMPT = """You are a senior financial analyst.
Compare the two SEC filings below and highlight meaningful year-over-year changes.

Focus on:
1. Revenue and profitability trends
2. Changes in risk factor language (new risks, removed risks, escalated risks)
3. Strategic direction shifts (acquisitions, new markets, product lines)
4. Capital allocation changes (buybacks, dividends, capex)
5. Management tone in MD&A

Filing 1 ({ticker} {type1} {year1}):
{context1}

Filing 2 ({ticker} {type2} {year2}):
{context2}

Provide your analysis with these sections:
FINANCIAL TRENDS: [quantitative comparison]
RISK EVOLUTION: [how risk language changed]
STRATEGIC SHIFTS: [business direction changes]
MANAGEMENT TONE: [language sentiment changes]
SUMMARY: [2-3 sentence overall assessment]"""


# ── Core LLM call ─────────────────────────────────────────────────────────────

def _call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not configured in .env")

    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": NVIDIA_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("NVIDIA API returned no choices.")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    if not content:
        raise RuntimeError("NVIDIA API returned empty content.")
    return content.strip()


# ── HyDE: hypothetical document embedding ────────────────────────────────────

def generate_hyde_document(question: str) -> str:
    """
    Generate a hypothetical SEC filing passage that would answer the question.

    HyDE (Hypothetical Document Embeddings) — Gao et al. 2022:
    Instead of embedding the raw question, embed this hypothetical answer.
    The resulting vector is much closer in semantic space to actual relevant
    passages in the filing, improving retrieval by 10–40% on financial text.
    """
    prompt = HYDE_GENERATION_PROMPT.format(question=question)
    return _call_llm(prompt, temperature=0.3, max_tokens=200)


# ── Financial Q&A ─────────────────────────────────────────────────────────────

def answer_financial_question(
    context_chunks: list[str],
    question: str,
    ticker: str = "",
    filing_type: str = "10-K",
    fiscal_year: int | str = "",
    metadatas: list[dict] | None = None,
) -> str:
    """Generate a cited, grounded answer to a financial question."""
    context = _build_context(context_chunks, metadatas)
    prompt = FINANCIAL_QA_PROMPT.format(
        ticker=ticker,
        filing_type=filing_type,
        fiscal_year=fiscal_year,
        context=context,
        question=question,
    )
    return _call_llm(prompt, temperature=0.1, max_tokens=600)


def _build_context(chunks: list[str], metadatas: list[dict] | None) -> str:
    if not metadatas:
        return "\n\n---\n\n".join(chunks)

    parts = []
    for text, meta in zip(chunks, metadatas):
        labels = []
        if meta.get("section_type"):
            labels.append(meta["section_type"].replace("_", " ").title())
        if meta.get("page_num"):
            labels.append(f"Page {meta['page_num']}")
        if meta.get("fiscal_year"):
            labels.append(f"FY{meta['fiscal_year']}")
        prefix = f"[{' | '.join(labels)}]" if labels else ""
        parts.append(f"{prefix}\n{text}".strip())
    return "\n\n---\n\n".join(parts)


# ── Structured metrics extraction ─────────────────────────────────────────────

def extract_financial_metrics(text: str) -> dict:
    """
    Extract structured financial KPIs from raw filing text.
    Returns a dict matching the METRICS_EXTRACTION_PROMPT schema.
    """
    # Send only the first ~4000 chars to keep tokens bounded
    snippet = text[:4000]
    prompt = METRICS_EXTRACTION_PROMPT.format(text=snippet)
    raw = _call_llm(prompt, temperature=0.0, max_tokens=500)

    # Parse JSON from response
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Return empty metrics on parse failure
    return {
        "revenue": None, "revenue_unit": None,
        "net_income": None, "net_income_unit": None,
        "eps_diluted": None, "gross_margin_pct": None,
        "operating_margin_pct": None, "total_assets": None,
        "total_assets_unit": None, "cash_and_equivalents": None,
        "cash_unit": None, "long_term_debt": None, "debt_unit": None,
        "r_and_d_expense": None, "r_and_d_unit": None, "employees": None,
        "fiscal_year": None, "key_risks": [], "strategic_highlights": [],
    }


# ── Year-over-year comparison ─────────────────────────────────────────────────

def compare_filings_yoy(
    chunks1: list[str],
    chunks2: list[str],
    ticker: str,
    year1: int,
    year2: int,
    type1: str = "10-K",
    type2: str = "10-K",
) -> dict:
    """
    Compare two sets of filing passages and return structured YoY analysis.
    """
    context1 = "\n\n---\n\n".join(chunks1[:5])
    context2 = "\n\n---\n\n".join(chunks2[:5])
    prompt = YOY_COMPARISON_PROMPT.format(
        ticker=ticker,
        type1=type1, year1=year1,
        type2=type2, year2=year2,
        context1=context1,
        context2=context2,
    )
    raw = _call_llm(prompt, temperature=0.1, max_tokens=800)
    return _parse_yoy_response(raw)


def _parse_yoy_response(raw: str) -> dict:
    sections = {
        "financial_trends": "",
        "risk_evolution": "",
        "strategic_shifts": "",
        "management_tone": "",
        "summary": "",
    }
    section_map = {
        "FINANCIAL TRENDS": "financial_trends",
        "RISK EVOLUTION": "risk_evolution",
        "STRATEGIC SHIFTS": "strategic_shifts",
        "MANAGEMENT TONE": "management_tone",
        "SUMMARY": "summary",
    }
    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        matched = False
        for label, key in section_map.items():
            if stripped.upper().startswith(label):
                current = key
                remainder = stripped[len(label):].lstrip(":").strip()
                if remainder:
                    sections[current] += remainder + "\n"
                matched = True
                break
        if not matched and current and stripped:
            sections[current] += stripped + "\n"

    if not any(sections.values()):
        sections["summary"] = raw.strip()
    return {k: v.strip() for k, v in sections.items()}


# ── Legacy compatibility (kept for backward compat with existing tests) ────────
