"""
Cross-encoder reranker.

Two-stage retrieval:
  Stage 1 → Dense vector search retrieves top-N candidates (fast, high recall).
  Stage 2 → Cross-encoder scores each (query, candidate) pair and returns top-K
             (slower, high precision — the cross-encoder reads both together).

Why cross-encoders beat bi-encoders for reranking:
  Bi-encoders embed query and doc independently — they miss fine-grained
  interactions.  Cross-encoders process the concatenated pair and learn much
  richer relevance signals at inference time.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  • Trained on MS-MARCO passage ranking (180M query-doc pairs)
  • 22 M parameters — fast on CPU
  • Returns a raw logit score; higher = more relevant
"""

from __future__ import annotations

from backend.config import RERANKER_MODEL_NAME

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder '{RERANKER_MODEL_NAME}'. "
                f"Ensure sentence-transformers >= 2.x is installed. Error: {exc}"
            )
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    text_key: str = "text",
) -> list[dict]:
    """
    Rerank a list of candidate dicts by relevance to *query*.

    Parameters
    ----------
    query      : the user's question
    candidates : list of dicts; each must have at least {text_key: str}
    top_k      : number of candidates to return
    text_key   : key in each dict that holds the text to score

    Returns
    -------
    top_k dicts sorted by cross-encoder score (descending),
    with a `rerank_score` key added to each.
    """
    if not candidates:
        return []

    model = _get_reranker()
    texts = [c[text_key] for c in candidates]
    pairs = [(query, t) for t in texts]

    scores = model.predict(pairs)

    scored = [
        {**c, "rerank_score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
