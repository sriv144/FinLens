import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import financial_parser, vector_store
from backend.services.financial_parser import FinancialChunk, ParsedDocument, INDEX_VERSION


def make_pdf_bytes(pages: list[str]) -> bytes:
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text)
    data = doc.tobytes()
    doc.close()
    return data


def build_chunk(
    chunk_id: str,
    text: str,
    chunk_kind: str,
    *,
    page_num: int = 1,
    parent_id: str | None = None,
    section_type: str = "risk_factors",
    ticker: str = "AAPL",
    company_name: str = "Apple Inc.",
    fiscal_year: int = 2024,
    filing_type: str = "10-K",
    filing_date: str = "2024-10-31",
) -> FinancialChunk:
    return FinancialChunk(
        chunk_id=chunk_id,
        text=text,
        chunk_kind=chunk_kind,
        section_type=section_type,
        page_num=page_num,
        parent_id=parent_id,
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        filing_type=filing_type,
        filing_date=filing_date,
    )


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_documents_routes(self) -> None:
        with patch(
            "backend.services.vector_store.list_documents",
            return_value=[
                {
                    "doc_id": "doc-1",
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "fiscal_year": 2024,
                    "filing_type": "10-K",
                    "filing_date": "2024-10-31",
                    "child_count": 8,
                    "table_count": 2,
                }
            ],
        ):
            health = self.client.get("/health")
            self.assertEqual(health.status_code, 200, health.text)
            self.assertEqual(health.json()["service"], "FinLens")

            documents = self.client.get("/documents")
            self.assertEqual(documents.status_code, 200, documents.text)
            body = documents.json()
            self.assertEqual(body["documents"][0]["doc_id"], "doc-1")
            self.assertEqual(body["documents"][0]["ticker"], "AAPL")

    def test_ingest_upload_returns_current_response_shape(self) -> None:
        parsed = ParsedDocument(
            ticker="AAPL",
            company_name="Apple Inc.",
            fiscal_year=2024,
            filing_type="10-K",
            filing_date="2024-10-31",
            parents=[build_chunk("p1", "Parent chunk", "parent")],
            children=[
                build_chunk("c1", "Risk factors mention supply chain concentration.", "child", parent_id="p1"),
                build_chunk("c2", "Management discusses AI features across devices.", "child", parent_id="p1"),
            ],
            tables=[build_chunk("t1", "Revenue | 394.3", "table", section_type="financial_statements")],
        )

        with (
            patch("backend.services.financial_parser.parse_pdf_filing", return_value=parsed),
            patch("backend.services.embedder.embed_texts", side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]),
            patch("backend.services.vector_store.store_parsed_document") as store_parsed_document,
            patch("backend.services.vector_store.store_document_metadata") as store_document_metadata,
        ):
            response = self.client.post(
                "/ingest/upload",
                files={"file": ("aapl.pdf", io.BytesIO(make_pdf_bytes(["Apple annual report"])), "application/pdf")},
                data={
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "fiscal_year": "2024",
                    "filing_type": "10-K",
                    "filing_date": "2024-10-31",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["ticker"], "AAPL")
        self.assertEqual(body["parent_count"], 1)
        self.assertEqual(body["child_count"], 2)
        self.assertEqual(body["table_count"], 1)
        self.assertIn("Successfully ingested AAPL 10-K 2024", body["message"])
        self.assertEqual(store_parsed_document.call_count, 3)
        store_document_metadata.assert_called_once()

    def test_ingest_upload_rejects_non_pdf(self) -> None:
        response = self.client.post(
            "/ingest/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
            data={
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "fiscal_year": "2024",
                "filing_type": "10-K",
                "filing_date": "2024-10-31",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_ingest_edgar_uses_current_route_and_parsing_flow(self) -> None:
        filing_meta = {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "fiscal_year": 2024,
            "form": "10-K",
            "filing_date": "2024-07-30",
        }
        parsed = ParsedDocument(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            fiscal_year=2024,
            filing_type="10-K",
            filing_date="2024-07-30",
            parents=[build_chunk("p1", "Parent chunk", "parent", ticker="MSFT", company_name="Microsoft Corporation")],
            children=[build_chunk("c1", "Cloud revenue grew strongly.", "child", parent_id="p1", ticker="MSFT", company_name="Microsoft Corporation")],
            tables=[],
        )

        with (
            patch("backend.services.edgar_client.fetch_filing_by_year", return_value=(filing_meta, "<html><body>Annual report</body></html>")),
            patch("backend.services.financial_parser.parse_html_filing", return_value=parsed),
            patch("backend.services.embedder.embed_texts", return_value=[[0.2, 0.3, 0.4]]),
            patch("backend.services.vector_store.store_parsed_document"),
            patch("backend.services.vector_store.store_document_metadata"),
        ):
            response = self.client.post(
                "/ingest/edgar",
                json={"ticker": "MSFT", "year": 2024, "form_type": "10-K"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["ticker"], "MSFT")

    def test_query_route_returns_sources_scores_and_stats(self) -> None:
        child_candidate = {
            "text": "Child risk chunk",
            "metadata": {
                "doc_id": "doc-1",
                "parent_id": "p1",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "section_type": "risk_factors",
                "page_num": 12,
                "filing_type": "10-K",
                "chunk_kind": "child",
            },
            "distance": 0.12,
        }
        table_candidate = {
            "text": "Revenue | 394.3",
            "metadata": {
                "doc_id": "doc-1",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "section_type": "financial_statements",
                "page_num": 88,
                "filing_type": "10-K",
                "chunk_kind": "table",
            },
            "distance": 0.08,
            "is_table": True,
        }
        parent_chunk = {
            "text": "Parent risk context with supply chain and manufacturing concentration disclosures.",
            "metadata": {
                "page_num": 12,
                "section_type": "risk_factors",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "filing_type": "10-K",
                "chunk_kind": "parent",
            },
        }
        reranked = [
            {**child_candidate, "rerank_score": 0.98},
            {**table_candidate, "rerank_score": 0.92},
        ]

        with (
            patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
            patch("backend.services.vector_store.query_children", return_value=[child_candidate]),
            patch("backend.services.vector_store.query_tables", return_value=[table_candidate]),
            patch("backend.services.reranker.rerank", return_value=reranked),
            patch("backend.services.vector_store.get_parents_by_ids", return_value=[parent_chunk]),
            patch("backend.services.llm.answer_financial_question", return_value="Grounded answer with citations."),
            patch(
                "backend.services.evaluator.evaluate_response",
                return_value={"faithfulness": 0.9, "relevance": 0.8, "quality_score": 0.86},
            ),
        ):
            response = self.client.post(
                "/query/",
                json={
                    "question": "What were revenue and the main risk factors?",
                    "doc_ids": ["doc-1"],
                    "use_hyde": False,
                    "top_k": 2,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["answer"], "Grounded answer with citations.")
        self.assertEqual(body["doc_ids_searched"], ["doc-1"])
        self.assertEqual(len(body["sources"]), 2)
        self.assertTrue(body["sources"][1]["is_table"])
        self.assertEqual(body["retrieval_stats"]["table_chunks_included"], 1)
        self.assertFalse(body["retrieval_stats"]["hyde_used"])
        self.assertEqual(body["evaluation"]["quality_score"], 0.86)

    def test_query_route_uses_hyde_when_enabled(self) -> None:
        candidate = {
            "text": "Risk factor candidate",
            "metadata": {
                "doc_id": "doc-1",
                "parent_id": "p1",
                "ticker": "NVDA",
                "fiscal_year": 2025,
                "section_type": "risk_factors",
                "page_num": 9,
                "filing_type": "10-K",
                "chunk_kind": "child",
            },
            "distance": 0.09,
        }

        with (
            patch("backend.services.llm.generate_hyde_document", return_value="Hypothetical filing answer."),
            patch("backend.services.embedder.embed_query", side_effect=[[0.9, 0.1], [0.8, 0.2]]),
            patch("backend.services.vector_store.query_children", return_value=[candidate]),
            patch("backend.services.vector_store.query_tables", return_value=[]),
            patch("backend.services.reranker.rerank", return_value=[{**candidate, "rerank_score": 0.95}]),
            patch("backend.services.vector_store.get_parents_by_ids", return_value=[]),
            patch("backend.services.llm.answer_financial_question", return_value="Answer from HyDE retrieval."),
            patch("backend.services.evaluator.evaluate_response", return_value={"faithfulness": 1.0, "relevance": 1.0, "quality_score": 1.0}),
        ):
            response = self.client.post(
                "/query/",
                json={
                    "question": "What AI risks did NVIDIA disclose?",
                    "doc_ids": ["doc-1"],
                    "use_hyde": True,
                    "top_k": 1,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["hyde_document"], "Hypothetical filing answer.")
        self.assertTrue(response.json()["retrieval_stats"]["hyde_used"])

    def test_compare_yoy_route_returns_structured_sections(self) -> None:
        metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2024,
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
            "child_count": 10,
            "table_count": 2,
        }
        comparison = {
            "financial_trends": "Revenue increased year over year.",
            "risk_evolution": "Supply chain language became more explicit.",
            "strategic_shifts": "AI features received more emphasis.",
            "management_tone": "Management remained confident but cautious.",
            "summary": "Overall performance improved with higher AI investment.",
        }

        with (
            patch(
                "backend.services.vector_store.get_document_metadata",
                side_effect=[metadata, {**metadata, "fiscal_year": 2025}],
            ),
            patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
            patch(
                "backend.services.vector_store.query_children",
                side_effect=[
                    [{"text": "2024 chunk", "metadata": metadata, "distance": 0.2}],
                    [{"text": "2025 chunk", "metadata": {**metadata, "fiscal_year": 2025}, "distance": 0.1}],
                ],
            ),
            patch("backend.services.llm.compare_filings_yoy", return_value=comparison),
        ):
            response = self.client.post(
                "/compare/yoy",
                json={"doc_id_1": "doc-2024", "doc_id_2": "doc-2025", "aspect": "AI strategy"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["ticker"], "AAPL")
        self.assertEqual(body["year_1"], 2024)
        self.assertEqual(body["year_2"], 2025)
        self.assertIn("Revenue increased", body["financial_trends"])

    def test_compare_yoy_surfaces_llm_failures_as_502(self) -> None:
        metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2024,
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
        }

        with (
            patch(
                "backend.services.vector_store.get_document_metadata",
                side_effect=[metadata, {**metadata, "fiscal_year": 2025}],
            ),
            patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
            patch(
                "backend.services.vector_store.query_children",
                side_effect=[
                    [{"text": "2024 chunk", "metadata": metadata, "distance": 0.2}],
                    [{"text": "2025 chunk", "metadata": {**metadata, "fiscal_year": 2025}, "distance": 0.1}],
                ],
            ),
            patch("backend.services.llm.compare_filings_yoy", side_effect=RuntimeError("provider denied")),
        ):
            response = self.client.post(
                "/compare/yoy",
                json={"doc_id_1": "doc-2024", "doc_id_2": "doc-2025"},
            )

        self.assertEqual(response.status_code, 502, response.text)
        self.assertIn("LLM comparison failed", response.json()["detail"])

    def test_metrics_extract_and_export_routes(self) -> None:
        metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2024,
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
            "child_count": 10,
            "table_count": 2,
        }
        metrics_payload = {
            "revenue": 394.3,
            "revenue_unit": "USD billions",
            "net_income": 99.8,
            "net_income_unit": "USD billions",
            "eps_diluted": 6.11,
            "gross_margin_pct": 45.2,
            "operating_margin_pct": 29.8,
            "total_assets": 352.6,
            "total_assets_unit": "USD billions",
            "cash_and_equivalents": 67.2,
            "cash_unit": "USD billions",
            "long_term_debt": 95.3,
            "debt_unit": "USD billions",
            "r_and_d_expense": 31.4,
            "r_and_d_unit": "USD billions",
            "employees": 161000,
            "key_risks": ["Supply chain concentration"],
            "strategic_highlights": ["Expanded on-device AI features"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "_metrics_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "doc-1.json"

            with (
                patch("backend.services.vector_store.get_document_metadata", return_value=metadata),
                patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
                patch(
                    "backend.services.vector_store.query_children",
                    return_value=[{"text": "Income statement excerpt", "metadata": metadata, "distance": 0.2}],
                ),
                patch(
                    "backend.services.vector_store.query_tables",
                    return_value=[{"text": "Revenue | 394.3", "metadata": metadata, "distance": 0.1, "is_table": True}],
                ),
                patch("backend.services.llm.extract_financial_metrics", return_value=metrics_payload),
                patch("backend.routers.metrics._CACHE_DIR", str(cache_dir)),
                patch("backend.routers.export.os.path.exists", side_effect=lambda p: Path(p) == cache_file),
                patch("backend.routers.export.open", create=True, side_effect=lambda p, *args, **kwargs: open(cache_file, *args, **kwargs)),
            ):
                extract_response = self.client.post("/metrics/extract/doc-1")
                self.assertEqual(extract_response.status_code, 200, extract_response.text)
                self.assertEqual(extract_response.json()["revenue"], 394.3)

                with open(cache_file, "w", encoding="utf-8") as handle:
                    json.dump({**metrics_payload, "doc_id": "doc-1", "ticker": "AAPL", "fiscal_year": 2024, "filing_type": "10-K"}, handle)

                export_response = self.client.get("/export/doc-1/excel")

        self.assertEqual(export_response.status_code, 200, export_response.text)
        self.assertIn("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", export_response.headers["content-type"])
        self.assertIn("FinLens_AAPL_10-K_2024.xlsx", export_response.headers["content-disposition"])

    def test_metrics_refresh_replaces_stale_empty_cache_from_sec_tables(self) -> None:
        metadata = {
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "fiscal_year": 2026,
            "filing_type": "10-K",
            "filing_date": "2026-02-25",
            "child_count": 10,
            "table_count": 2,
        }
        stale_metrics = {
            "doc_id": "doc-1",
            "ticker": "NVDA",
            "fiscal_year": 2026,
            "filing_type": "10-K",
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
        table_text = (
            "Year Ended | Jan 25, 2026 | Jan 26, 2025 | ($ in millions, except per share data) "
            "Revenue | $ | 215,938 | | $ | 130,497 | Gross margin | 71.1 | % | 75.0 | % "
            "Operating income | $ | 130,387 | | $ | 81,453 | Net income | $ | 120,067 | | $ | 72,880 "
            "Net income per diluted share | $ | 4.90 | | $ | 2.94 "
            "Cash and cash equivalents | $ | 10,605 | | $ | 8,589 | Total assets | $ | 206,803 | | $ | 111,601 "
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "_metrics_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "doc-1.json"
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump(stale_metrics, handle)

            with (
                patch("backend.services.vector_store.get_document_metadata", return_value=metadata),
                patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
                patch("backend.services.vector_store.query_children", return_value=[]),
                patch(
                    "backend.services.vector_store.query_tables",
                    return_value=[{"text": table_text, "metadata": metadata, "distance": 0.1, "is_table": True}],
                ),
                patch("backend.services.llm.extract_financial_metrics", return_value={}),
                patch("backend.routers.metrics._CACHE_DIR", str(cache_dir)),
            ):
                refresh_response = self.client.post("/metrics/extract/doc-1")
                cached_response = self.client.get("/metrics/doc-1")

        self.assertEqual(refresh_response.status_code, 200, refresh_response.text)
        body = refresh_response.json()
        self.assertEqual(body["revenue"], 215938.0)
        self.assertEqual(body["revenue_unit"], "USD millions")
        self.assertEqual(body["net_income"], 120067.0)
        self.assertEqual(body["eps_diluted"], 4.90)
        self.assertEqual(body["gross_margin_pct"], 71.1)
        self.assertEqual(body["total_assets"], 206803.0)
        self.assertEqual(body["cash_and_equivalents"], 10605.0)
        self.assertEqual(cached_response.json()["revenue"], 215938.0)
    def test_metrics_extract_handles_apple_total_net_sales_tables(self) -> None:
        metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2025,
            "filing_type": "10-K",
            "filing_date": "2025-10-31",
            "child_count": 10,
            "table_count": 2,
        }
        table_text = (
            "Deferred revenue | $ | 2,953 | "
            "Products | $ | 307,003 | Services | 109,158 | Total net sales | $ | 416,161 | "
            "Gross margin | $ | 195,201 | Operating income | $ | 133,050 | "
            "Net income | $ | 112,010 | Diluted | $ | 7.46 | "
            "Cash and cash equivalents | $ | 35,934 | Total assets | $ | 359,241 | "
            "Term debt | $ | 12,350 | Non-current liabilities: | Term debt | $ | 78,328 | "
            "Capitalized research and development | 15,041 | "
            "Research and development | $ | 34,550 |"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "_metrics_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            with (
                patch("backend.services.vector_store.get_document_metadata", return_value=metadata),
                patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
                patch("backend.services.vector_store.query_children", return_value=[]),
                patch(
                    "backend.services.vector_store.query_tables",
                    return_value=[{"text": table_text, "metadata": metadata, "distance": 0.1, "is_table": True}],
                ),
                patch("backend.services.llm.extract_financial_metrics", return_value={}),
                patch("backend.routers.metrics._CACHE_DIR", str(cache_dir)),
            ):
                response = self.client.post("/metrics/extract/doc-apple")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["revenue"], 416161.0)
        self.assertEqual(body["revenue_unit"], "USD millions")
        self.assertEqual(body["net_income"], 112010.0)
        self.assertEqual(body["eps_diluted"], 7.46)
        self.assertEqual(body["gross_margin_pct"], 46.9)
        self.assertEqual(body["operating_margin_pct"], 32.0)
        self.assertEqual(body["total_assets"], 359241.0)
        self.assertEqual(body["cash_and_equivalents"], 35934.0)
        self.assertEqual(body["long_term_debt"], 78328.0)
        self.assertEqual(body["r_and_d_expense"], 34550.0)

    def test_metrics_extract_surfaces_llm_failures_as_502(self) -> None:
        metadata = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2024,
            "filing_type": "10-K",
            "filing_date": "2024-10-31",
            "child_count": 10,
            "table_count": 2,
        }

        with (
            patch("backend.services.vector_store.get_document_metadata", return_value=metadata),
            patch("backend.services.embedder.embed_query", return_value=[0.1, 0.2, 0.3]),
            patch(
                "backend.services.vector_store.query_children",
                return_value=[{"text": "Income statement excerpt", "metadata": metadata, "distance": 0.2}],
            ),
            patch("backend.services.vector_store.query_tables", return_value=[]),
            patch("backend.services.llm.extract_financial_metrics", side_effect=RuntimeError("provider denied")),
        ):
            response = self.client.post("/metrics/extract/doc-1")

        self.assertEqual(response.status_code, 502, response.text)
        self.assertIn("LLM metrics extraction failed", response.json()["detail"])

    def test_parse_pdf_filing_builds_current_chunk_model(self) -> None:
        pdf_bytes = make_pdf_bytes(
            [
                "Item 1A. Risk Factors\nSupply chain concentration and geopolitical risks may affect operations.",
                "Item 8. Financial Statements\nRevenue 394.3\nNet income 99.8",
            ]
        )

        parsed = financial_parser.parse_pdf_filing(
            pdf_bytes,
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "fiscal_year": 2024,
                "form": "10-K",
                "filing_date": "2024-10-31",
            },
        )

        self.assertGreater(len(parsed.parents), 0)
        self.assertGreater(len(parsed.children), 0)
        self.assertTrue(all(chunk.chunk_kind == "parent" for chunk in parsed.parents))
        self.assertTrue(all(chunk.chunk_kind == "child" for chunk in parsed.children))
        self.assertTrue(
            all(
                chunk.section_type in {"risk_factors", "other", "financial_statements"}
                for chunk in parsed.parents + parsed.children + parsed.tables
            )
        )

    def test_store_document_metadata_and_list_documents_match_current_shape(self) -> None:
        recorded_name = None

        class FakeCollection:
            def __init__(self) -> None:
                self.rows: dict[str, dict] = {}

            def upsert(self, *, ids, documents, metadatas, embeddings) -> None:
                for idx, meta in zip(ids, metadatas):
                    self.rows[idx] = meta

            def get(self, ids=None, include=None):
                if ids:
                    metas = [self.rows[idx] for idx in ids if idx in self.rows]
                    return {"metadatas": metas}
                all_ids = list(self.rows)
                all_metas = [self.rows[idx] for idx in all_ids]
                return {"ids": all_ids, "metadatas": all_metas}

        class FakeClient:
            def __init__(self) -> None:
                self.collection = FakeCollection()

            def get_or_create_collection(self, name: str):
                nonlocal recorded_name
                recorded_name = name
                return self.collection

            def get_collection(self, name: str):
                return self.collection

        fake_client = FakeClient()
        with patch.object(vector_store, "get_client", return_value=fake_client):
            vector_store.store_document_metadata(
                "doc-1",
                ticker="AAPL",
                company_name="Apple Inc.",
                fiscal_year=2024,
                filing_type="10-K",
                filing_date="2024-10-31",
                parent_count=3,
                child_count=10,
                table_count=2,
                index_version=INDEX_VERSION,
            )
            vector_store.store_document_metadata(
                "doc-2",
                ticker="MSFT",
                company_name="Microsoft Corporation",
                fiscal_year=2023,
                filing_type="10-K",
                filing_date="2023-07-30",
                parent_count=2,
                child_count=8,
                table_count=1,
                index_version="legacy_v0",
            )

            listed = vector_store.list_documents()

        self.assertEqual(recorded_name, vector_store.DOCUMENT_METADATA_COLLECTION)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["doc_id"], "doc-1")
        self.assertEqual(listed[0]["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()





