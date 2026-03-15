<div align="center">

# RAG Decision Intelligence Engine

**Production-grade evidence retrieval and decision analysis system powered by Hybrid RAG, ML reliability scoring, NLI contradiction detection, and LLM reasoning.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

The RAG Decision Intelligence Engine is a multi-stage AI pipeline that answers comparative and strategic questions by retrieving evidence from a knowledge base, scoring its reliability and credibility, detecting contradictions between sources, applying policy-based decision logic, and generating a structured, auditable decision report with LLM reasoning.

It is designed for production use: all models are loaded once at startup, all configs are environment-driven, all components are independently testable, and the full stack runs in Docker.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│       Decision Parser        │  Regex-based option extraction (A vs B)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Hybrid Retrieval        │  FAISS (dense) + BM25 (sparse)
│                              │  fused via Reciprocal Rank Fusion (k=60)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Cross-Encoder Reranker    │  ms-marco-MiniLM-L-6-v2
│                              │  Sigmoid-normalised relevance scores
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Reliability Scoring       │  XGBoost (6-feature sklearn Pipeline)
│    + Credibility Blending    │  final = 0.5×sim + 0.3×rel + 0.2×cred
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Contradiction Detection    │  NLI — nli-deberta-v3-small
│                              │  Pairwise O(n²), capped at 30 pairs
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Decision Policy Engine     │  Rule-based signal evaluation
│                              │  Contradiction / score margin / evidence count
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Self-Reflection Agent       │  Post-decision evidence audit
│                              │  Credibility distribution + retrieval gaps
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      LLM Reasoning           │  Ollama (Llama3) via LangChain
│                              │  Rule-based fallback when Ollama unavailable
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Decision Report         │  Pydantic schema — JSON + Streamlit UI
└─────────────────────────────┘
```

---

## Key Features

- **Hybrid Retrieval** — combines FAISS dense vector search with BM25 sparse keyword search, fused via Reciprocal Rank Fusion. BM25 corpus is automatically seeded from persisted FAISS metadata on restart — no separate persistence layer required.
- **ML Reliability Model** — XGBoost classifier trained on 6 evidence features: similarity score, document length, citation presence, publication year, source credibility, and chunk index. Tracked via MLflow.
- **NLI Contradiction Detection** — pairwise entailment scoring across retrieved evidence using a fine-tuned DeBERTa NLI cross-encoder. Flags contradicting sources before a decision is made.
- **Decision Policy Engine** — rule-based layer that evaluates contradiction signals, score margins, and evidence count before selecting a recommendation. Returns `Inconclusive` when evidence is insufficient or ambiguous.
- **Self-Reflection Agent** — post-decision validator that audits evidence completeness, credibility distribution, and retrieval gaps. Applies a confidence penalty when quality is low.
- **Structured Decision Reports** — fully typed Pydantic schema with per-option evidence breakdown, contradiction details, policy reasoning, reflection flags, and LLM narrative.
- **Production Ready** — singleton model loading, structlog JSON logging, FastAPI lifespan management, Docker multi-stage builds, Render Blueprint deployment.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| RAG Framework | LangChain |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | FAISS `IndexFlatIP` with L2 normalisation |
| Keyword Search | BM25Okapi (`rank-bm25`) |
| Reranker | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| NLI Model | CrossEncoder `nli-deberta-v3-small` |
| LLM Runtime | Ollama — Llama3 |
| Reliability Model | XGBoost / RandomForest (sklearn Pipeline) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | PostgreSQL (SQLAlchemy) |
| Config | pydantic-settings |
| Logging | structlog (structured JSON) |
| Deployment | Docker, Docker Compose, Render |

---

## Project Structure

```
rag_decision_engine/
├── config/
│   ├── settings.py                # Pydantic-settings — all env-driven config
│   └── logging_config.py          # Structlog JSON logging setup
│
├── data_pipeline/
│   ├── ingest.py                  # End-to-end ingestion orchestrator
│   ├── preprocess.py              # Text normalisation + metadata extraction
│   └── chunking.py                # RecursiveCharacterTextSplitter (LangChain)
│
├── retrieval/
│   ├── vector_retriever.py        # FAISS dense retrieval
│   ├── bm25_retriever.py          # BM25 sparse retrieval
│   ├── hybrid_retriever.py        # RRF fusion + BM25 seeding from FAISS
│   └── reranker.py                # Cross-encoder reranking + sigmoid norm
│
├── models/
│   ├── train_model.py             # XGBoost training + MLflow tracking
│   └── predict.py                 # ReliabilityPredictor singleton
│
├── services/
│   ├── db.py                      # SQLAlchemy engine factory
│   ├── reliability_service.py     # Evidence scoring + credibility blending
│   ├── contradiction_service.py   # NLI pairwise contradiction detection
│   ├── decision_policy.py         # Policy engine — rule-based signal evaluation
│   ├── reflection_service.py      # Self-reflection evidence audit agent
│   └── decision_service.py        # Full pipeline orchestrator
│
├── api/
│   └── server.py                  # FastAPI — /decision, /ingest_documents, /health, /metrics
│
├── dashboard/
│   └── app.py                     # Streamlit UI — decision analysis, ingestion, metrics
│
├── evaluation/
│   └── evaluate.py                # Recall@k, Precision@k, MRR, decision accuracy
│
└── tests/
    ├── test_preprocess.py
    ├── test_chunking.py
    ├── test_reliability.py
    └── test_hybrid_retriever.py
```

---

## Quickstart — Local

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Ollama](https://ollama.com) installed locally

### 1. Clone and install

```bash
git clone https://github.com/Sahojit/AIOps-Pipeline-RCA.git
cd AIOps-Pipeline-RCA

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set POSTGRES_USER, POSTGRES_HOST, OLLAMA_BASE_URL
```

### 3. Start infrastructure

```bash
docker compose up postgres mlflow -d
ollama pull llama3
```

### 4. Train the reliability model

```bash
python -m rag_decision_engine.models.train_model
```

### 5. Start API and dashboard

```bash
# Terminal 1
python -m rag_decision_engine.api.server
# → http://localhost:8000
# → http://localhost:8000/docs

# Terminal 2
streamlit run rag_decision_engine/dashboard/app.py
# → http://localhost:8501
```

---

## Docker Deployment

```bash
# Build and start full stack
docker compose up --build -d

# Pull LLM model into the Ollama container
docker exec rag_ollama ollama pull llama3

# Verify
curl http://localhost:8000/health
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8080 |
| Swagger UI | http://localhost:8080/docs |
| Streamlit | http://localhost:8090 |
| MLflow | http://localhost:6001 |
| PostgreSQL | localhost:5440 |

---

## Cloud Deployment — Render

The repo includes a `render.yaml` Blueprint. All services are pre-configured to pull from Docker Hub (`docker.io/sahojit/`).

**One-click deploy:**

1. Go to [render.com](https://render.com) → **New → Blueprint**
2. Connect this repository
3. Render provisions PostgreSQL, Ollama (with persistent 10 GB disk), MLflow, API, and Dashboard automatically

| Service | Plan | Notes |
|---|---|---|
| PostgreSQL | Free | Managed database |
| `rag-ollama` | Standard | Pulls llama3 on first start, cached to disk |
| `rag-mlflow` | Starter | File-based tracking |
| `rag-api` | Standard | 2 GB RAM required for ML models |
| `rag-dashboard` | Starter | Streamlit frontend |

---

## API Reference

### `POST /decision`

Run the full decision pipeline.

**Request**
```json
{
  "query": "Should I use XGBoost or Random Forest for tabular datasets?"
}
```

**Response**
```json
{
  "query": "Should I use XGBoost or Random Forest for tabular datasets?",
  "options": [
    {
      "option": "XGBoost",
      "supporting_documents": 5,
      "reliability_score": 0.82,
      "source_credibility": 0.91,
      "final_evidence_score": 0.87,
      "top_evidence_snippets": ["XGBoost consistently outperforms..."]
    },
    {
      "option": "Random Forest",
      "supporting_documents": 3,
      "reliability_score": 0.63,
      "source_credibility": 0.78,
      "final_evidence_score": 0.71,
      "top_evidence_snippets": ["Random Forest is more interpretable..."]
    }
  ],
  "contradiction_detected": false,
  "contradiction_count": 0,
  "recommended_option": "XGBoost",
  "recommendation_confidence": 0.76,
  "policy_reason": "XGBoost leads by margin 0.16 with 5 supporting documents.",
  "reflection_flag": false,
  "reflection_reason": "",
  "reasoning": "Based on the retrieved evidence...",
  "latency_ms": 843.2
}
```

### `POST /ingest_documents`

Add documents to the knowledge base.

```json
{
  "texts": ["XGBoost is a gradient boosting framework..."],
  "file_paths": ["/data/research_paper.pdf"],
  "force_reingest": false
}
```

### `GET /health`

Returns `{"status": "ok"}` — used by Docker and Render health checks.

### `GET /metrics`

Returns operational counters: `request_count`, `decision_count`, `ingest_count`, `error_count`, `avg_latency_ms`.

---

## Decision Policy Rules

The `DecisionPolicyEngine` evaluates four signals before selecting a recommendation:

| Condition | Outcome |
|---|---|
| Contradiction detected AND max contradiction score > 0.6 | `Inconclusive` — conflicting evidence |
| Score margin between top two options < 0.05 | `Inconclusive` — insufficient differentiation |
| Supporting documents < 3 | Recommendation flagged as low-confidence |
| Otherwise | Highest scoring option recommended |

---

## Evidence Scoring Formula

```
final_score = 0.5 × similarity_score
            + 0.3 × reliability_score
            + 0.2 × source_credibility
```

Configurable via environment variables: `WEIGHT_SIMILARITY`, `WEIGHT_RELIABILITY`, `WEIGHT_CREDIBILITY`.

---

## Source Credibility Scale

| Source Type | Score |
|---|---|
| Research Paper | 0.95 |
| Official Documentation | 0.90 |
| Technical Blog | 0.70 |
| News Article | 0.60 |
| Forum / Community Post | 0.40 |
| Unknown | 0.50 |

Override via `SOURCE_CREDIBILITY_MAP` in `.env`.

---

## Reliability Model Features

The XGBoost reliability classifier is trained on six evidence features:

| Feature | Description |
|---|---|
| `similarity_score` | Cross-encoder relevance score (sigmoid-normalised) |
| `doc_length_tokens` | Word count of the chunk |
| `has_citations` | Binary — presence of citation markers |
| `publication_year_norm` | Normalised recency score |
| `source_credibility` | Look-up table score by source type |
| `chunk_index_norm` | Position of chunk within original document |

Training accuracy: **0.99** — cross-validation F1: **0.87** on 3,000 synthetic samples.
Track experiments via MLflow at `http://localhost:6001`.

---

## Evaluation

```bash
python -m rag_decision_engine.evaluation.evaluate
```

```
Retrieval  Recall@10:       0.84
Retrieval  Precision@10:    0.72
Retrieval  MRR:             0.69
Decision   Accuracy:        0.75
Decision   Avg Confidence:  0.68
Decision   Contradiction %: 33.3%
Decision   Avg Latency ms:  1240
```

---

## Tests

```bash
pytest rag_decision_engine/tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `rag_decision_engine` | Database name |
| `POSTGRES_USER` | — | Database user |
| `POSTGRES_PASSWORD` | — | Database password (optional) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking URI |
| `RETRIEVAL_TOP_K` | `20` | Candidates retrieved before reranking |
| `RERANK_TOP_K` | `10` | Top-k after reranking |
| `WEIGHT_SIMILARITY` | `0.5` | Evidence formula weight |
| `WEIGHT_RELIABILITY` | `0.3` | Evidence formula weight |
| `WEIGHT_CREDIBILITY` | `0.2` | Evidence formula weight |
| `CONTRADICTION_THRESHOLD` | `0.5` | NLI contradiction detection threshold |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## License

MIT
