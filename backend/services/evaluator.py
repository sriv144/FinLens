"""
RAG answer evaluation.

Implements two lightweight, LLM-based evaluation metrics inspired by RAGAS:

  faithfulness_score  (0.0 – 1.0)
    "Is every factual claim in the answer actually supported by the context?"
    1.0 = fully grounded, 0.0 = entirely fabricated.

  answer_relevance_score  (0.0 – 1.0)
    "Does the answer actually address the question asked?"
    1.0 = perfectly on-topic, 0.0 = completely off-topic.

Why this matters for a resume project:
  Most RAG demos skip evaluation entirely.  Showing faithfulness scores in
  the UI signals production-level thinking: the system knows when it might
  be hallucinating and communicates that uncertainty to the user.

Implementation note:
  We use a single lightweight LLM call with a structured 0-10 integer prompt
  and parse the integer from the response.  This avoids RAGAS as a dependency
  while keeping the same conceptual framework.
"""

from __future__ import annotations

import re

from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_LLM_MODEL
import requests

_FAITHFULNESS_PROMPT = """You are a strict RAG evaluation judge.

Your task: decide whether the ANSWER is fully supported by the CONTEXT.

Rules:
- Score 10 if EVERY factual claim in the answer is directly supported by the context.
- Score 0 if the answer contains facts not present in the context (hallucination).
- Intermediate scores reflect partial support.
- Output ONLY a single integer from 0 to 10. No explanation.

CONTEXT:
{context}

ANSWER:
{answer}

SCORE (0-10):"""

_RELEVANCE_PROMPT = """You are a strict RAG evaluation judge.

Your task: decide whether the ANSWER actually addresses the QUESTION.

Rules:
- Score 10 if the answer directly and completely answers the question.
- Score 0 if the answer is completely off-topic or refuses to answer without reason.
- Intermediate scores reflect partial relevance.
- Output ONLY a single integer from 0 to 10. No explanation.

QUESTION:
{question}

ANSWER:
{answer}

SCORE (0-10):"""


def _call_llm(prompt: str) -> str:
    if not NVIDIA_API_KEY:
        return "5"
    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": NVIDIA_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4,
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return content.strip()
    except Exception:
        return "5"


def _parse_score(raw: str) -> float:
    match = re.search(r"\b([0-9]|10)\b", raw)
    if match:
        return round(int(match.group(1)) / 10, 1)
    return 0.5


def compute_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Return a faithfulness score in [0.0, 1.0].
    1.0 = answer is fully grounded in context.
    """
    context = "\n\n---\n\n".join(context_chunks[:5])  # cap tokens
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = _call_llm(prompt)
    return _parse_score(raw)


def compute_relevance(question: str, answer: str) -> float:
    """
    Return an answer-relevance score in [0.0, 1.0].
    1.0 = answer directly addresses the question.
    """
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw = _call_llm(prompt)
    return _parse_score(raw)


def evaluate_response(
    question: str,
    answer: str,
    context_chunks: list[str],
) -> dict:
    """
    Run both evaluations and return a dict with faithfulness, relevance, and
    a composite quality score.
    """
    faithfulness = compute_faithfulness(answer, context_chunks)
    relevance = compute_relevance(question, answer)
    # Weighted composite: faithfulness is more critical in RAG
    quality = round(0.6 * faithfulness + 0.4 * relevance, 2)
    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "quality_score": quality,
    }
