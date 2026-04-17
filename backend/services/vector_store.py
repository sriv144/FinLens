"""
ChromaDB vector store for FinLens.

Collection layout per document
────────────────────────────────────────────────────────
  {doc_id}__children   — small child chunks (what gets embedded & retrieved)
  {doc_id}__parents    — large parent chunks (returned as context after retrieval)
  {doc_id}__tables     — financial table chunks (routed for numeric queries)
  doc_metadata         — document-level metadata (ticker, year, filing_type, …)

Design rationale
────────────────
Parent-child retrieval decouples precision (small child embeddings → accurate
retrieval) from context quality (full parent chunks → coherent LLM context).
This yields the best of both worlds without the complexity of a separate
reranking index.

Table chunks are stored separately so the query router can inject them when it
detects numeric / financial-metric questions (revenue, EPS, margins, …).
"""

from __future__ import annotations

import chromadb

from backend.config import CHROMA_DB_PATH
from backend.services.financial_parser import INDEX_VERSION, FinancialChunk

_client = None
DOCUMENT_METADATA_COLLECTION = "doc_metadata"

_CHILDREN_SUFFIX = "__children"
_PARENTS_SUFFIX  = "__parents"
_TABLES_SUFFIX   = "__tables"


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


# ── Collection name helpers ────────────────────────────────────────────────────

def _col(doc_id: str, suffix: str) -> str:
    return f"{doc_id}{suffix}"


# ── Store ──────────────────────────────────────────────────────────────────────

def store_parsed_document(
    doc_id: str,
    chunks: list[FinancialChunk],
    embeddings: list[list[float]],
    kind: str,  # "child" | "parent" | "table"
) -> None:
    """Store a list of FinancialChunks with their embeddings into the correct collection."""
    suffix_map = {
        "child":  _CHILDREN_SUFFIX,
        "parent": _PARENTS_SUFFIX,
        "table":  _TABLES_SUFFIX,
    }
    suffix = suffix_map[kind]
    client = get_client()
    col = client.get_or_create_collection(
        name=_col(doc_id, suffix),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [_chunk_to_metadata(doc_id, c) for c in chunks]

    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def _chunk_to_metadata(doc_id: str, chunk: FinancialChunk) -> dict:
    meta = {
        "doc_id": doc_id,
        "chunk_kind": chunk.chunk_kind,
        "section_type": chunk.section_type,
        "page_num": chunk.page_num,
        "ticker": chunk.ticker,
        "company_name": chunk.company_name,
        "fiscal_year": chunk.fiscal_year,
        "filing_type": chunk.filing_type,
        "filing_date": chunk.filing_date,
        "index_version": INDEX_VERSION,
    }
    if chunk.parent_id:
        meta["parent_id"] = chunk.parent_id
    return meta


# ── Query children → expand to parents ────────────────────────────────────────

def query_children(
    doc_id: str,
    query_embedding: list[float],
    top_k: int = 40,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Dense vector search over child chunks.
    Returns list of dicts with keys: text, metadata, distance.
    """
    client = get_client()
    try:
        col = client.get_collection(_col(doc_id, _CHILDREN_SUFFIX))
    except Exception:
        return []

    where = {"section_type": section_filter} if section_filter else None
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    result = col.query(**kwargs)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]


def get_parents_by_ids(doc_id: str, parent_ids: list[str]) -> list[dict]:
    """
    Retrieve parent chunks by their IDs (for context expansion after child retrieval).
    """
    if not parent_ids:
        return []
    client = get_client()
    try:
        col = client.get_collection(_col(doc_id, _PARENTS_SUFFIX))
        result = col.get(ids=parent_ids, include=["documents", "metadatas"])
    except Exception:
        return []

    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]


def query_tables(
    doc_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Dense search over financial table chunks."""
    client = get_client()
    try:
        col = client.get_collection(_col(doc_id, _TABLES_SUFFIX))
        if col.count() == 0:
            return []
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return [{"text": d, "metadata": m, "distance": dist, "is_table": True}
            for d, m, dist in zip(docs, metas, dists)]


# ── Document metadata ──────────────────────────────────────────────────────────

def store_document_metadata(
    doc_id: str,
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    filing_type: str,
    filing_date: str,
    parent_count: int,
    child_count: int,
    table_count: int,
    # Legacy compat
    total_pages: int = 0,
    filename: str = "",
    index_version: str = INDEX_VERSION,
    structure_unit_count: int = 0,
    passage_unit_count: int = 0,
) -> None:
    client = get_client()
    col = client.get_or_create_collection(name=DOCUMENT_METADATA_COLLECTION)
    label = filename or f"{ticker}_{filing_type}_{fiscal_year}"
    col.upsert(
        ids=[doc_id],
        documents=[label],
        metadatas=[{
            "ticker": ticker,
            "company_name": company_name,
            "fiscal_year": fiscal_year,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "parent_count": parent_count,
            "child_count": child_count,
            "table_count": table_count,
            "total_pages": total_pages,
            "filename": label,
            "index_version": index_version,
        }],
        embeddings=[[0.0]],
    )


def get_document_metadata(doc_id: str) -> dict | None:
    client = get_client()
    try:
        col = client.get_collection(name=DOCUMENT_METADATA_COLLECTION)
        result = col.get(ids=[doc_id], include=["metadatas"])
        metas = result.get("metadatas", [])
        if metas:
            return metas[0]
    except Exception:
        pass
    return None


def list_documents() -> list[dict]:
    """Return all ingested documents with their metadata."""
    client = get_client()
    try:
        col = client.get_collection(name=DOCUMENT_METADATA_COLLECTION)
        result = col.get(include=["metadatas"])
    except Exception:
        return []

    ids = result.get("ids", [])
    metas = result.get("metadatas", [])
    docs = []
    for doc_id, meta in zip(ids, metas):
        if (meta or {}).get("index_version") == INDEX_VERSION:
            docs.append({"doc_id": doc_id, **(meta or {})})
    return sorted(docs, key=lambda d: (d.get("ticker", ""), -d.get("fiscal_year", 0)))


def delete_document(doc_id: str) -> None:
    client = get_client()
    for suffix in (_CHILDREN_SUFFIX, _PARENTS_SUFFIX, _TABLES_SUFFIX):
        try:
            client.delete_collection(_col(doc_id, suffix))
        except Exception:
            pass
    # Remove from doc_metadata
    try:
        col = client.get_collection(DOCUMENT_METADATA_COLLECTION)
        col.delete(ids=[doc_id])
    except Exception:
        pass


# ── Legacy compatibility ────────────────────────────────────────────────────────
# Old routes (upload.py, compare.py) still call these names.

PASSAGE_SUFFIX   = "__children"
STRUCTURE_SUFFIX = "__parents"


def query_collection(doc_id: str, query_embedding: list[float], top_k: int) -> dict:
    """Legacy shim — routes to query_children."""
    results = query_children(doc_id, query_embedding, top_k)
    return {
        "documents": [[r["text"] for r in results]],
        "metadatas": [[r["metadata"] for r in results]],
        "distances": [[r["distance"] for r in results]],
    }


def get_chunks(doc_id: str) -> list[dict]:
    """Legacy shim — returns child chunks."""
    client = get_client()
    try:
        col = client.get_collection(_col(doc_id, _CHILDREN_SUFFIX))
        result = col.get(include=["documents", "metadatas"])
        return [
            {"text": d, "metadata": m}
            for d, m in zip(result["documents"], result["metadatas"])
        ]
    except Exception:
        return []


def get_structure_units(doc_id: str) -> list[dict]:
    """Legacy shim — returns parent chunks."""
    client = get_client()
    try:
        col = client.get_collection(_col(doc_id, _PARENTS_SUFFIX))
        result = col.get(include=["documents", "metadatas"])
        return [
            {"text": d, "metadata": m}
            for d, m in zip(result["documents"], result["metadatas"])
        ]
    except Exception:
        return []


def store_passage_units(doc_id: str, units: list[dict], embeddings: list[list[float]]) -> None:
    """Legacy shim for old upload router."""
    pass


def store_structure_units(doc_id: str, units: list[dict], embeddings: list[list[float]]) -> None:
    """Legacy shim for old upload router."""
    pass
