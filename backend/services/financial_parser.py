"""
Financial document parser with table-aware chunking and parent-child indexing.

Handles both:
  • HTML filings from SEC EDGAR  (primary path)
  • PDF uploads                  (fallback path via PyMuPDF)

Key design decisions
────────────────────
Parent-Child chunking
  Parent chunks (~1 500 chars) carry full context; child chunks (~300 chars)
  are what gets embedded and retrieved. On retrieval we expand child → parent.
  This gives the precision of small chunks with the coherence of large context.

Table-aware extraction
  Financial tables (income statement, balance sheet, cash flow) are extracted
  as structured text and stored as dedicated "table" chunks, separately tagged
  so the query layer can route numeric questions directly to them.

Section tagging
  SEC 10-K sections are detected by keyword patterns and tagged with a
  `section_type` label (business, risk_factors, mda, financial_statements,
  notes, other).  This enables metadata-filtered retrieval: e.g. retrieve only
  from `risk_factors` when the analyst asks "what are the key risks?".

INDEX_VERSION
  Bump this string whenever the schema changes so old documents are re-ingested.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from backend.config import PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP

INDEX_VERSION = "finlens_v1"

SectionType = Literal[
    "business", "risk_factors", "mda", "financial_statements", "notes", "other"
]

# ── Section keyword detection ─────────────────────────────────────────────────

_SECTION_PATTERNS: list[tuple[SectionType, re.Pattern]] = [
    ("business",            re.compile(r"\bitem\s*1\b(?!\s*a)", re.I)),
    ("risk_factors",        re.compile(r"\bitem\s*1a\b|\brisk factors\b", re.I)),
    ("mda",                 re.compile(r"\bitem\s*7\b|management.{0,20}discussion", re.I)),
    ("financial_statements",re.compile(r"\bitem\s*8\b|financial\s+statements", re.I)),
    ("notes",               re.compile(r"\bnotes\s+to\s+(consolidated\s+)?financial", re.I)),
]

_TABLE_INDICATORS = re.compile(
    r"(revenue|net income|earnings|diluted|total assets|shareholders|equity"
    r"|operating|cash flow|fiscal|quarter|annual|Q[1-4]\b)",
    re.I,
)


def _detect_section(text: str) -> SectionType:
    snippet = text[:200]
    for section_type, pattern in _SECTION_PATTERNS:
        if pattern.search(snippet):
            return section_type
    return "other"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FinancialChunk:
    chunk_id: str
    text: str
    chunk_kind: Literal["parent", "child", "table"]
    section_type: SectionType
    page_num: int
    parent_id: str | None      # set for child chunks
    ticker: str
    company_name: str
    fiscal_year: int
    filing_type: str
    filing_date: str
    char_start: int = 0


@dataclass
class ParsedDocument:
    ticker: str
    company_name: str
    fiscal_year: int
    filing_type: str
    filing_date: str
    parents: list[FinancialChunk] = field(default_factory=list)
    children: list[FinancialChunk] = field(default_factory=list)
    tables: list[FinancialChunk] = field(default_factory=list)
    full_text: str = ""


# ── HTML parser ───────────────────────────────────────────────────────────────

def _parse_html(html: str) -> tuple[list[tuple[int, str]], list[str]]:
    """
    Return (page_segments, table_texts).

    page_segments: list of (page_num, text_block) – we treat each <div>/<p>
                   cluster as a synthetic "page".
    table_texts:   list of table text representations.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError(
            "beautifulsoup4 is required for HTML parsing. "
            "Run: pip install beautifulsoup4 lxml"
        )

    soup = BeautifulSoup(html, "lxml")

    # Remove script / style / header / footer noise
    for tag in soup(["script", "style", "head", "nav", "footer"]):
        tag.decompose()

    # Extract tables first
    tables: list[str] = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        table_text = "\n".join(rows)
        if _TABLE_INDICATORS.search(table_text) and len(table_text) > 80:
            tables.append(table_text)
        table.decompose()  # remove from soup so we don't double-count

    # Extract body text in order, treating top-level blocks as synthetic pages
    body = soup.find("body") or soup
    blocks = body.find_all(
        ["p", "div", "h1", "h2", "h3", "h4", "span", "li"], recursive=False
    )
    if not blocks:
        blocks = body.find_all(["p", "div", "h1", "h2", "h3", "h4", "span"])

    segments: list[tuple[int, str]] = []
    buffer = []
    synthetic_page = 1
    char_count = 0
    PAGE_THRESHOLD = 3000  # chars before we bump synthetic page

    for block in blocks:
        text = block.get_text(separator=" ", strip=True)
        if not text or len(text) < 20:
            continue
        buffer.append(text)
        char_count += len(text)
        if char_count >= PAGE_THRESHOLD:
            segments.append((synthetic_page, " ".join(buffer)))
            buffer = []
            char_count = 0
            synthetic_page += 1

    if buffer:
        segments.append((synthetic_page, " ".join(buffer)))

    if not segments:
        # Last resort: raw text
        raw = soup.get_text(separator="\n", strip=True)
        segments = [(1, raw)]

    return segments, tables


# ── PDF parser ────────────────────────────────────────────────────────────────

def _parse_pdf(pdf_bytes: bytes) -> tuple[list[tuple[int, str]], list[str]]:
    """Return (page_segments, table_texts) from PDF bytes."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF is required: pip install PyMuPDF")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    segments: list[tuple[int, str]] = []
    tables: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            segments.append((page_num, text.strip()))

        # Table extraction (PyMuPDF >= 1.23)
        try:
            for tbl in page.find_tables():
                rows = []
                for row in tbl.extract():
                    cells = [str(c) if c else "" for c in row]
                    rows.append(" | ".join(cells))
                table_text = "\n".join(rows)
                if _TABLE_INDICATORS.search(table_text) and len(table_text) > 80:
                    tables.append(f"[Page {page_num}]\n{table_text}")
        except Exception:
            pass

    doc.close()
    return segments, tables


# ── Chunking utilities ────────────────────────────────────────────────────────

def _split_into_parents(text: str, size: int = PARENT_CHUNK_SIZE) -> list[str]:
    """Split text into overlapping parent chunks by character count."""
    chunks = []
    start = 0
    overlap = size // 10
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _split_parent_to_children(
    parent_text: str,
    size: int = CHILD_CHUNK_SIZE,
    overlap: int = CHILD_CHUNK_OVERLAP,
) -> list[str]:
    """Split a parent chunk into smaller child chunks."""
    children = []
    start = 0
    while start < len(parent_text):
        end = start + size
        child = parent_text[start:end].strip()
        if len(child) > 30:
            children.append(child)
        start = end - overlap
    return children


# ── Main parse entry points ───────────────────────────────────────────────────

def parse_html_filing(
    html: str,
    filing_meta: dict,
) -> ParsedDocument:
    """
    Parse an EDGAR HTML filing into a ParsedDocument with parent/child/table chunks.
    """
    ticker = filing_meta.get("ticker", "UNKNOWN")
    company_name = filing_meta.get("company_name", ticker)
    fiscal_year = filing_meta.get("fiscal_year", 0)
    filing_type = filing_meta.get("form", "10-K")
    filing_date = filing_meta.get("filing_date", "")

    segments, raw_tables = _parse_html(html)

    doc = ParsedDocument(
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        filing_type=filing_type,
        filing_date=filing_date,
    )

    full_parts: list[str] = []

    for page_num, page_text in segments:
        full_parts.append(page_text)
        section = _detect_section(page_text)
        parent_texts = _split_into_parents(page_text)

        for pidx, parent_text in enumerate(parent_texts):
            parent_id = f"p_{uuid.uuid4().hex[:12]}"
            parent_chunk = FinancialChunk(
                chunk_id=parent_id,
                text=parent_text,
                chunk_kind="parent",
                section_type=section,
                page_num=page_num,
                parent_id=None,
                ticker=ticker,
                company_name=company_name,
                fiscal_year=fiscal_year,
                filing_type=filing_type,
                filing_date=filing_date,
            )
            doc.parents.append(parent_chunk)

            child_texts = _split_parent_to_children(parent_text)
            for cidx, child_text in enumerate(child_texts):
                child_id = f"c_{uuid.uuid4().hex[:12]}"
                child_chunk = FinancialChunk(
                    chunk_id=child_id,
                    text=child_text,
                    chunk_kind="child",
                    section_type=section,
                    page_num=page_num,
                    parent_id=parent_id,
                    ticker=ticker,
                    company_name=company_name,
                    fiscal_year=fiscal_year,
                    filing_type=filing_type,
                    filing_date=filing_date,
                )
                doc.children.append(child_chunk)

    # Table chunks
    for tidx, table_text in enumerate(raw_tables):
        section = _detect_section(table_text)
        table_id = f"t_{uuid.uuid4().hex[:12]}"
        table_chunk = FinancialChunk(
            chunk_id=table_id,
            text=table_text[:2000],  # cap very large tables
            chunk_kind="table",
            section_type=section,
            page_num=0,
            parent_id=None,
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            filing_type=filing_type,
            filing_date=filing_date,
        )
        doc.tables.append(table_chunk)

    doc.full_text = "\n\n".join(full_parts)
    return doc


def parse_pdf_filing(
    pdf_bytes: bytes,
    filing_meta: dict,
) -> ParsedDocument:
    """
    Parse a PDF filing into a ParsedDocument.
    Same output schema as parse_html_filing.
    """
    ticker = filing_meta.get("ticker", "UNKNOWN")
    company_name = filing_meta.get("company_name", ticker)
    fiscal_year = filing_meta.get("fiscal_year", 0)
    filing_type = filing_meta.get("form", "10-K")
    filing_date = filing_meta.get("filing_date", "")

    segments, raw_tables = _parse_pdf(pdf_bytes)

    doc = ParsedDocument(
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        filing_type=filing_type,
        filing_date=filing_date,
    )

    full_parts: list[str] = []

    for page_num, page_text in segments:
        full_parts.append(page_text)
        section = _detect_section(page_text)
        parent_texts = _split_into_parents(page_text)

        for parent_text in parent_texts:
            parent_id = f"p_{uuid.uuid4().hex[:12]}"
            doc.parents.append(FinancialChunk(
                chunk_id=parent_id,
                text=parent_text,
                chunk_kind="parent",
                section_type=section,
                page_num=page_num,
                parent_id=None,
                ticker=ticker,
                company_name=company_name,
                fiscal_year=fiscal_year,
                filing_type=filing_type,
                filing_date=filing_date,
            ))
            for child_text in _split_parent_to_children(parent_text):
                child_id = f"c_{uuid.uuid4().hex[:12]}"
                doc.children.append(FinancialChunk(
                    chunk_id=child_id,
                    text=child_text,
                    chunk_kind="child",
                    section_type=section,
                    page_num=page_num,
                    parent_id=parent_id,
                    ticker=ticker,
                    company_name=company_name,
                    fiscal_year=fiscal_year,
                    filing_type=filing_type,
                    filing_date=filing_date,
                ))

    for table_text in raw_tables:
        table_id = f"t_{uuid.uuid4().hex[:12]}"
        doc.tables.append(FinancialChunk(
            chunk_id=table_id,
            text=table_text[:2000],
            chunk_kind="table",
            section_type=_detect_section(table_text),
            page_num=0,
            parent_id=None,
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
            filing_type=filing_type,
            filing_date=filing_date,
        ))

    doc.full_text = "\n\n".join(full_parts)
    return doc
