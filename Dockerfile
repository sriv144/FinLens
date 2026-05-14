# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by lxml, PyMuPDF, and sentence-transformers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Backend on 8000, Streamlit on 8501. docker-compose overrides CMD per service.
EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
