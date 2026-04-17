"""
FinLens — Financial Document Intelligence Platform
Streamlit frontend
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "300"))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinLens — Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    border-right: 1px solid #2d4a6e;
}
[data-testid="stSidebar"] * {
    color: #e8eaed !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #8ab4d4 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Metric cards ────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.metric-unit {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* ── Answer box ──────────────────────────────────────────── */
.answer-box {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
    color: #0f172a;
    font-size: 0.97rem;
    line-height: 1.7;
    margin: 12px 0;
}

/* ── Source card ─────────────────────────────────────────── */
.source-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.55;
}
.source-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    text-transform: uppercase;
}
.table-badge {
    background: #fef3c7;
    color: #92400e;
}

/* ── Score chips ─────────────────────────────────────────── */
.score-chip {
    display: inline-block;
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
}
.score-high   { background: #dcfce7; color: #166534; }
.score-medium { background: #fef9c3; color: #854d0e; }
.score-low    { background: #fee2e2; color: #991b1b; }

/* ── HyDE snippet ────────────────────────────────────────── */
.hyde-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.83rem;
    color: #14532d;
    font-style: italic;
}

/* ── Section header ──────────────────────────────────────── */
.section-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
}

/* ── Document pill ───────────────────────────────────────── */
.doc-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}

/* ── Logo ────────────────────────────────────────────────── */
.finlens-logo {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.finlens-logo span { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ── State helpers ─────────────────────────────────────────────────────────────

def get_documents() -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-medium"
    return "score-low"


def _fmt_metric(value, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value >= 1:
        formatted = f"{value:,.1f}"
    else:
        formatted = str(value)
    return f"{formatted} {unit}".strip() if unit else formatted


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="finlens-logo">Fin<span>Lens</span> 📊</div>', unsafe_allow_html=True)
    st.caption("Financial Document Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Ingest Filing", "🔍 Ask a Question", "📈 Compare Years", "📋 Metrics Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()

    # Loaded filings
    docs = get_documents()
    if docs:
        st.markdown("**Loaded Filings**")
        for d in docs:
            ticker = d.get("ticker", "?")
            year = d.get("fiscal_year", "?")
            ftype = d.get("filing_type", "")
            st.markdown(
                f'<div class="doc-pill">{ticker} {ftype} {year}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No filings loaded yet. Start by ingesting one.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INGEST
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Ingest Filing":
    st.title("Ingest SEC Filing")
    st.caption("Load a filing from SEC EDGAR by ticker, or upload a PDF directly.")

    tab1, tab2 = st.tabs(["📡 Fetch from EDGAR", "📎 Upload PDF"])

    # ── EDGAR fetch ────────────────────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker_input = st.text_input("Ticker Symbol", placeholder="e.g. AAPL, MSFT, NVDA")
        with col2:
            form_type = st.selectbox("Filing Type", ["10-K", "10-Q"])
        with col3:
            year_input = st.number_input(
                "Fiscal Year (0 = latest)", min_value=0, max_value=2030, value=0
            )

        if st.button("🚀 Fetch & Ingest", type="primary", use_container_width=True):
            if not ticker_input.strip():
                st.error("Please enter a ticker symbol.")
            else:
                with st.spinner(f"Fetching {ticker_input.upper()} {form_type} from SEC EDGAR…"):
                    payload = {
                        "ticker": ticker_input.strip(),
                        "form_type": form_type,
                        "year": int(year_input) if year_input > 0 else None,
                    }
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/ingest/edgar",
                            json=payload,
                            timeout=TIMEOUT,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(f"✅ {data['message']}")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Doc ID", data["doc_id"])
                        col_b.metric("Fiscal Year", data["fiscal_year"])
                        col_c.metric("Child Chunks", f"{data['child_count']:,}")
                        col_d.metric("Table Chunks", f"{data['table_count']:,}")
                        st.info("Refresh the page to see this filing in the sidebar.")
                    except requests.exceptions.HTTPError as e:
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(f"EDGAR error: {detail or str(e)}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        # Available filings preview
        if ticker_input.strip():
            with st.expander("📋 Preview available filings for this ticker"):
                try:
                    r = requests.get(
                        f"{BACKEND_URL}/ingest/filing-list/{ticker_input.upper()}",
                        params={"form_type": form_type, "max_filings": 5},
                        timeout=20,
                    )
                    r.raise_for_status()
                    filings = r.json().get("filings", [])
                    for f in filings:
                        st.markdown(
                            f"**{f['form']} {f['fiscal_year']}** — "
                            f"Filed {f['filing_date']} — "
                            f"`{f['accession_number']}`"
                        )
                except Exception:
                    pass

    # ── PDF upload ─────────────────────────────────────────────────────────────
    with tab2:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        col1, col2, col3 = st.columns(3)
        with col1:
            pdf_ticker = st.text_input("Ticker", key="pdf_ticker", placeholder="AAPL")
        with col2:
            pdf_year = st.number_input("Fiscal Year", min_value=2000, max_value=2030, value=2024, key="pdf_year")
        with col3:
            pdf_type = st.selectbox("Type", ["10-K", "10-Q", "Other"], key="pdf_type")
        pdf_company = st.text_input("Company Name (optional)", key="pdf_company")
        pdf_date = st.text_input("Filing Date (YYYY-MM-DD, optional)", key="pdf_date")

        if st.button("📤 Upload & Ingest", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please select a PDF file.")
            elif not pdf_ticker.strip():
                st.error("Please enter a ticker symbol.")
            else:
                with st.spinner("Parsing and indexing PDF…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/ingest/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                            data={
                                "ticker": pdf_ticker.strip().upper(),
                                "company_name": pdf_company.strip(),
                                "fiscal_year": str(pdf_year),
                                "filing_type": pdf_type,
                                "filing_date": pdf_date.strip(),
                            },
                            timeout=TIMEOUT,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(f"✅ {data['message']}")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Doc ID", data["doc_id"])
                        col_b.metric("Child Chunks", f"{data['child_count']:,}")
                        col_c.metric("Table Chunks", f"{data['table_count']:,}")
                    except Exception as e:
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(f"Upload failed: {detail or str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ASK A QUESTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Ask a Question":
    st.title("Ask a Financial Question")
    st.caption("Powered by HyDE query expansion + cross-encoder reranking + NVIDIA LLM")

    docs = get_documents()
    if not docs:
        st.warning("No filings loaded. Go to **🏠 Ingest Filing** to load one first.")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        doc_options = {
            f"{d.get('ticker','?')} {d.get('filing_type','?')} {d.get('fiscal_year','?')} (ID: {d.get('doc_id','?')})": d.get("doc_id")
            for d in docs
        }
        selected_labels = st.multiselect(
            "Select filing(s) to query",
            options=list(doc_options.keys()),
            default=[list(doc_options.keys())[0]] if doc_options else [],
        )
        selected_doc_ids = [doc_options[l] for l in selected_labels if l in doc_options]

    with col2:
        section_filter = st.selectbox(
            "Section filter",
            ["All sections", "risk_factors", "mda", "financial_statements", "business", "notes"],
        )
        use_hyde = st.toggle("HyDE expansion", value=True, help="Generates a hypothetical answer to improve retrieval accuracy")

    # ── Example questions ─────────────────────────────────────────────────────
    st.caption("**Try asking:**")
    example_qs = [
        "What are the top 3 risk factors mentioned?",
        "What was the annual revenue and net income?",
        "How does management describe their AI strategy?",
        "What acquisitions or divestitures occurred?",
    ]
    q_cols = st.columns(len(example_qs))
    for col, q in zip(q_cols, example_qs):
        if col.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["prefilled_q"] = q

    question = st.text_area(
        "Your question",
        value=st.session_state.get("prefilled_q", ""),
        height=100,
        placeholder="What were the key risk factors disclosed in this filing?",
    )

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not selected_doc_ids:
            st.error("Select at least one filing to query.")
        elif not question.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Retrieving relevant passages and generating answer…"):
                payload = {
                    "question": question.strip(),
                    "doc_ids": selected_doc_ids,
                    "section_filter": None if section_filter == "All sections" else section_filter,
                    "use_hyde": use_hyde,
                    "top_k": 5,
                }
                try:
                    resp = requests.post(f"{BACKEND_URL}/query/", json=payload, timeout=TIMEOUT)
                    resp.raise_for_status()
                    result = resp.json()

                    # ── HyDE snippet ──────────────────────────────────────────
                    if result.get("hyde_document") and use_hyde:
                        with st.expander("🧠 HyDE hypothetical passage (used for retrieval)"):
                            st.markdown(
                                f'<div class="hyde-box">{result["hyde_document"]}</div>',
                                unsafe_allow_html=True,
                            )

                    # ── Answer ────────────────────────────────────────────────
                    st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Evaluation scores ─────────────────────────────────────
                    ev = result.get("evaluation")
                    if ev:
                        faith = ev.get("faithfulness", 0)
                        relev = ev.get("relevance", 0)
                        qual  = ev.get("quality_score", 0)
                        score_html = (
                            f'<div style="margin-top:12px">'
                            f'<span class="score-chip {_score_class(faith)}">Faithfulness {faith:.0%}</span>'
                            f'<span class="score-chip {_score_class(relev)}">Relevance {relev:.0%}</span>'
                            f'<span class="score-chip {_score_class(qual)}">Quality {qual:.0%}</span>'
                            f'</div>'
                        )
                        st.markdown(score_html, unsafe_allow_html=True)

                    # ── Retrieval stats ───────────────────────────────────────
                    stats = result.get("retrieval_stats", {})
                    st.caption(
                        f"Retrieved {stats.get('total_candidates', 0)} candidates → "
                        f"Reranked to {stats.get('after_rerank', 0)} → "
                        f"Used {stats.get('context_chunks', 0)} context chunks"
                        + (" · 📊 Table chunks included" if stats.get("table_chunks_included") else "")
                    )

                    # ── Sources ───────────────────────────────────────────────
                    sources = result.get("sources", [])
                    if sources:
                        st.markdown('<div class="section-header" style="margin-top:20px">Source Excerpts</div>', unsafe_allow_html=True)
                        for i, src in enumerate(sources, 1):
                            section = src.get("section_type", "").replace("_", " ").title()
                            ticker = src.get("ticker", "")
                            year = src.get("fiscal_year", "")
                            page = src.get("page_num", 0)
                            score = src.get("rerank_score")
                            is_table = src.get("is_table", False)

                            badge_html = ""
                            if section:
                                badge_html += f'<span class="source-badge">{section}</span>'
                            if is_table:
                                badge_html += '<span class="source-badge table-badge">📊 Table</span>'
                            if ticker:
                                badge_html += f'<span class="source-badge">{ticker} {year}</span>'
                            if page:
                                badge_html += f'<span class="source-badge">p.{page}</span>'
                            if score is not None:
                                badge_html += f'<span class="source-badge" style="background:#e0e7ff;color:#3730a3">Score {score:.2f}</span>'

                            text = src.get("text", "")[:400]
                            st.markdown(
                                f'<div class="source-card">'
                                f'{badge_html}<br/><br/>{text}…'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                except requests.exceptions.HTTPError as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Query failed: {detail or str(e)}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — YEAR-OVER-YEAR COMPARE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Compare Years":
    st.title("Year-over-Year Comparison")
    st.caption("Compare two filings to identify strategic shifts, evolving risks, and financial trends.")

    docs = get_documents()
    if len(docs) < 2:
        st.warning("Load at least **two filings** to use comparison. Go to 🏠 Ingest Filing.")
        st.stop()

    doc_options = {
        f"{d.get('ticker','?')} {d.get('filing_type','?')} {d.get('fiscal_year','?')}": d.get("doc_id")
        for d in docs
    }
    labels = list(doc_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        label1 = st.selectbox("Filing 1 (older)", labels, index=0)
    with col2:
        label2 = st.selectbox("Filing 2 (newer)", labels, index=min(1, len(labels)-1))

    aspect = st.text_input(
        "Focus area (optional)",
        placeholder="e.g. 'AI strategy', 'supply chain risks', 'revenue growth'"
    )

    if st.button("🔄 Compare", type="primary", use_container_width=True):
        if label1 == label2:
            st.error("Select two different filings to compare.")
        else:
            with st.spinner("Comparing filings…"):
                payload = {
                    "doc_id_1": doc_options[label1],
                    "doc_id_2": doc_options[label2],
                    "aspect": aspect.strip() or None,
                }
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/compare/yoy", json=payload, timeout=TIMEOUT
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    st.subheader(
                        f"{data.get('ticker','?')} · {data.get('year_1','?')} vs {data.get('year_2','?')}"
                    )

                    sections = [
                        ("📉 Financial Trends", data.get("financial_trends", "")),
                        ("⚠️ Risk Evolution", data.get("risk_evolution", "")),
                        ("🎯 Strategic Shifts", data.get("strategic_shifts", "")),
                        ("💬 Management Tone", data.get("management_tone", "")),
                    ]

                    for title, content in sections:
                        if content:
                            st.markdown(f"**{title}**")
                            st.markdown(
                                f'<div class="answer-box">{content}</div>',
                                unsafe_allow_html=True,
                            )
                            st.divider()

                    if data.get("summary"):
                        st.info(f"**Summary:** {data['summary']}")

                except Exception as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Comparison failed: {detail or str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — METRICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Metrics Dashboard":
    st.title("Financial Metrics Dashboard")
    st.caption("Structured KPI extraction from ingested filings.")

    docs = get_documents()
    if not docs:
        st.warning("No filings loaded. Go to 🏠 Ingest Filing first.")
        st.stop()

    doc_options = {
        f"{d.get('ticker','?')} {d.get('filing_type','?')} {d.get('fiscal_year','?')}": d.get("doc_id")
        for d in docs
    }
    selected_label = st.selectbox("Select a filing", list(doc_options.keys()))
    selected_id = doc_options[selected_label]

    col_btn, col_dl = st.columns([1, 1])
    extract_clicked = col_btn.button("📊 Extract / Refresh Metrics", type="primary")
    col_dl.link_button(
        "⬇️ Download Excel Report",
        f"{BACKEND_URL}/export/{selected_id}/excel",
    )

    if extract_clicked:
        with st.spinner("Extracting metrics with LLM…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/metrics/extract/{selected_id}", timeout=TIMEOUT
                )
                resp.raise_for_status()
                metrics = resp.json()
                st.session_state[f"metrics_{selected_id}"] = metrics
                st.success("Metrics extracted successfully.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    # Auto-fetch cached metrics on load
    if f"metrics_{selected_id}" not in st.session_state:
        try:
            resp = requests.get(f"{BACKEND_URL}/metrics/{selected_id}", timeout=30)
            if resp.ok:
                st.session_state[f"metrics_{selected_id}"] = resp.json()
        except Exception:
            pass

    metrics = st.session_state.get(f"metrics_{selected_id}")
    if not metrics:
        st.info("Click 'Extract / Refresh Metrics' to analyse this filing.")
        st.stop()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    st.subheader(f"Key Metrics — {metrics.get('ticker','')} {metrics.get('filing_type','')} {metrics.get('fiscal_year','')}")

    def _metric_card(label: str, value, unit: str = "") -> str:
        val_str = _fmt_metric(value, unit)
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val_str}</div>'
            f'</div>'
        )

    kpi_rows = [
        [
            ("Revenue", metrics.get("revenue"), metrics.get("revenue_unit") or ""),
            ("Net Income", metrics.get("net_income"), metrics.get("net_income_unit") or ""),
            ("EPS (Diluted)", metrics.get("eps_diluted"), "USD"),
        ],
        [
            ("Gross Margin", metrics.get("gross_margin_pct"), "%"),
            ("Operating Margin", metrics.get("operating_margin_pct"), "%"),
            ("Total Assets", metrics.get("total_assets"), metrics.get("total_assets_unit") or ""),
        ],
        [
            ("Cash & Equivalents", metrics.get("cash_and_equivalents"), metrics.get("cash_unit") or ""),
            ("Long-Term Debt", metrics.get("long_term_debt"), metrics.get("debt_unit") or ""),
            ("R&D Expense", metrics.get("r_and_d_expense"), metrics.get("r_and_d_unit") or ""),
        ],
    ]

    for row in kpi_rows:
        cols = st.columns(3)
        for col, (label, value, unit) in zip(cols, row):
            col.markdown(_metric_card(label, value, unit), unsafe_allow_html=True)
        st.write("")

    # ── Employees ──────────────────────────────────────────────────────────────
    if metrics.get("employees"):
        st.markdown(
            f'<div class="metric-card" style="width:33%">'
            f'<div class="metric-label">Employees</div>'
            f'<div class="metric-value">{int(metrics["employees"]):,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Key risks ──────────────────────────────────────────────────────────────
    if metrics.get("key_risks"):
        st.subheader("⚠️ Key Risks Identified")
        for risk in metrics["key_risks"]:
            st.markdown(f"- {risk}")

    # ── Strategic highlights ────────────────────────────────────────────────────
    if metrics.get("strategic_highlights"):
        st.subheader("🎯 Strategic Highlights")
        for highlight in metrics["strategic_highlights"]:
            st.markdown(f"- {highlight}")
