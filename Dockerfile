FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY . .

# Pre-bake the embedding model into the image so cold starts never re-download from HuggingFace.
# snapshot_download fetches the repo files into the proper HF cache tree without loading
# PyTorch — no OOM risk during Docker build.
ENV HF_HOME=/app/hf_cache
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')"

RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/data /app/logs /app/models && \
    chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    HF_HOME=/app/hf_cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1
CMD ["sh", "-c", "uvicorn rag_decision_engine.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
