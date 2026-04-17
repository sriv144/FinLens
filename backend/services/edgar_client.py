"""
SEC EDGAR API client.

Fetches 10-K / 10-Q filings by ticker symbol using public EDGAR endpoints.
No API key required — SEC mandates a User-Agent header identifying the caller.

Key endpoints used:
  Company tickers  → https://www.sec.gov/files/company_tickers.json
  Submissions      → https://data.sec.gov/submissions/CIK{cik:010d}.json
  Filing archive   → https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

import requests

from backend.config import EDGAR_USER_AGENT, EDGAR_BASE_URL, EDGAR_ARCHIVES_URL

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"})

# Polite delay between EDGAR requests (SEC fair-use policy)
_REQUEST_DELAY = 0.12  # seconds


def _get(url: str, **kwargs) -> requests.Response:
    time.sleep(_REQUEST_DELAY)
    resp = _SESSION.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


# ── Ticker → CIK ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_tickers() -> dict[str, int]:
    """Return a {ticker_upper: cik_int} mapping from the EDGAR master list."""
    data: dict[str, Any] = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {entry["ticker"].upper(): entry["cik_str"] for entry in data.values()}


def get_cik(ticker: str) -> int:
    """Resolve a ticker symbol to its SEC CIK (integer)."""
    mapping = _load_tickers()
    ticker_up = ticker.upper().strip()
    if ticker_up not in mapping:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR. "
                         f"Check the symbol and try again.")
    return mapping[ticker_up]


# ── Submissions (filing index) ────────────────────────────────────────────────

def get_submissions(cik: int) -> dict:
    """Fetch the full submissions JSON for a company (filing history + metadata)."""
    url = f"{EDGAR_BASE_URL}/submissions/CIK{cik:010d}.json"
    return _get(url).json()


def get_company_info(ticker: str) -> dict:
    """Return basic company metadata from EDGAR."""
    cik = get_cik(ticker)
    subs = get_submissions(cik)
    return {
        "cik": cik,
        "ticker": ticker.upper(),
        "name": subs.get("name", "Unknown"),
        "sic": subs.get("sic", ""),
        "state": subs.get("stateOfIncorporation", ""),
    }


# ── Filing list ───────────────────────────────────────────────────────────────

def list_filings(
    ticker: str,
    form_type: str = "10-K",
    max_filings: int = 5,
) -> list[dict]:
    """
    Return a list of filing metadata dicts for the given ticker and form type.

    Each dict has: accession_number, filing_date, fiscal_year_end,
                   primary_document, form, cik, company_name.
    """
    cik = get_cik(ticker)
    subs = get_submissions(cik)
    company_name = subs.get("name", ticker.upper())
    recent = subs.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    fiscal_year_ends = recent.get("fiscalYearEnd", [None] * len(forms))

    results = []
    for form, acc, date, doc, fye in zip(
        forms, accessions, filing_dates, primary_docs,
        fiscal_year_ends if fiscal_year_ends else [None] * len(forms),
    ):
        if form.upper() != form_type.upper():
            continue
        results.append({
            "form": form,
            "accession_number": acc,
            "filing_date": date,
            "fiscal_year": int(date[:4]),
            "fiscal_year_end": fye or "",
            "primary_document": doc,
            "cik": cik,
            "company_name": company_name,
            "ticker": ticker.upper(),
        })
        if len(results) >= max_filings:
            break

    if not results:
        raise ValueError(
            f"No {form_type} filings found for '{ticker}'. "
            f"Try a different form type (e.g. '10-Q') or check the ticker."
        )
    return results


# ── Filing content download ───────────────────────────────────────────────────

def _accession_no_dashes(accession_number: str) -> str:
    return accession_number.replace("-", "")


def download_filing(filing: dict) -> tuple[str, str]:
    """
    Download the primary filing document and return (html_or_text, content_type).

    For 10-K / 10-Q filings the primary document is usually an HTML file.
    Falls back to downloading the filing index and picking the largest HTML doc.
    """
    cik = filing["cik"]
    acc = _accession_no_dashes(filing["accession_number"])
    primary_doc = filing.get("primary_document", "")

    # Try primary document first
    if primary_doc:
        url = f"{EDGAR_ARCHIVES_URL}/{cik}/{acc}/{primary_doc}"
        try:
            resp = _get(url)
            ctype = resp.headers.get("content-type", "text/html")
            return resp.text, ctype
        except Exception:
            pass

    # Fallback: read the index JSON and pick the biggest htm file
    index_url = f"{EDGAR_ARCHIVES_URL}/{cik}/{acc}/{acc}-index.json"
    try:
        index = _get(index_url).json()
        items = index.get("directory", {}).get("item", [])
        htm_items = [
            i for i in items
            if i.get("name", "").lower().endswith((".htm", ".html"))
            and "index" not in i.get("name", "").lower()
        ]
        htm_items.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
        if htm_items:
            doc_name = htm_items[0]["name"]
            url = f"{EDGAR_ARCHIVES_URL}/{cik}/{acc}/{doc_name}"
            resp = _get(url)
            return resp.text, "text/html"
    except Exception:
        pass

    raise RuntimeError(
        f"Could not download filing {filing['accession_number']} for {filing['ticker']}."
    )


# ── Convenience: fetch latest 10-K for a ticker ───────────────────────────────

def fetch_latest_10k(ticker: str) -> tuple[dict, str]:
    """
    Return (filing_metadata, html_content) for the most recent 10-K.
    """
    filings = list_filings(ticker, form_type="10-K", max_filings=1)
    filing = filings[0]
    html, _ = download_filing(filing)
    return filing, html


def fetch_filing_by_year(ticker: str, year: int, form_type: str = "10-K") -> tuple[dict, str]:
    """
    Return (filing_metadata, html_content) for a specific fiscal year.
    """
    filings = list_filings(ticker, form_type=form_type, max_filings=10)
    for filing in filings:
        if filing["fiscal_year"] == year:
            html, _ = download_filing(filing)
            return filing, html
    available_years = [f["fiscal_year"] for f in filings]
    raise ValueError(
        f"No {form_type} found for {ticker} fiscal year {year}. "
        f"Available years: {available_years}"
    )
