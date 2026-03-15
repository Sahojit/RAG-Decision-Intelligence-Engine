# RAG Decision Intelligence Engine

A production-grade applied AI system that retrieves evidence from documents,
evaluates the reliability and credibility of that evidence, detects contradictions
between sources, and generates structured decision reports.

---

## System Architecture

```
User Query
    │
    ▼
Decision Parser           ← extracts decision options from query
    │
    ▼
Hybrid Retrieval          ← FAISS (dense) + BM25 (sparse) + RRF fusion
    │
    ▼
Cross-Encoder Reranker    ← re-scores candidate documents for relevance
    │
    ▼
Evidence Extraction       ← scored chunks with metadata
    │
    ▼
Reliability Model         ← XGBoost / RandomForest classifier (sklearn)
    │
    ▼
Source Credibility        ← look-up table (research paper > blog > forum)
    │
    ▼
Contradiction Detection   ← NLI cross-encoder (ENTAILMENT / CONTRADICTION / NEUTRAL)
    │
    ▼
Decision Reasoning        ← LLM reasoning via Ollama (Llama3 / Mistral)
    │
    ▼
Decision Report           ← structured JSON + Streamlit dashboard
```

---

## Example Output

```
Decision Report
───────────────────────────────────────────────
Query: Should I use XGBoost or Random Forest for tabular datasets?

Option: XGBoost
  Supporting Documents : 5
  Reliability Score    : 0.82
  Source Credibility   : 0.91
  Evidence Score       : 0.87

Option: Random Forest
  Supporting Documents : 3
  Reliability Score    : 0.63
  Source Credibility   : 0.78
  Evidence Score       : 0.71

Contradictory Evidence  : YES (2 pairs)
Recommended Decision    : XGBoost
Confidence              : 0.76
───────────────────────────────────────────────
```

---

## Project Structure

```
rag_decision_engine/
├── config/
│   ├── settings.py          # all env-driven config (pydantic-settings)
│   └── logging_config.py    # structured JSON logging (structlog)
│
├── data_pipeline/
│   ├── ingest.py            # end-to-end ingestion orchestrator
│   ├── preprocess.py        # text cleaning & metadata extraction
│   └── chunking.py          # recursive character chunking (LangChain)
│
├── retrieval/
│   ├── vector_retriever.py  # FAISS + SentenceTransformers
│   ├── bm25_retriever.py    # BM25 (rank-bm25)
│   ├── hybrid_retriever.py  # RRF fusion of dense + sparse
│   └── reranker.py          # cross-encoder reranker
│
├── models/
│   ├── train_model.py       # XGBoost / RF training + MLflow tracking
│   └── predict.py           # inference singleton
│
├── services/
│   ├── db.py                # SQLAlchemy engine factory
│   ├── reliability_service.py   # ML scoring + credibility blending
│   ├── contradiction_service.py # NLI contradiction detection
│   └── decision_service.py      # full pipeline orchestration
│
├── api/
│   └── server.py            # FastAPI app (POST /decision, POST /ingest_documents, …)
│
├── dashboard/
│   └── app.py               # Streamlit UI
│
├── evaluation/
│   └── evaluate.py          # Recall@k, MRR, decision accuracy
│
└── tests/
    ├── test_preprocess.py
    ├── test_chunking.py
    ├── test_reliability.py
    └── test_hybrid_retriever.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| RAG Framework | LangChain |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| NLI Model | CrossEncoder (`nli-deberta-v3-small`) |
| LLM Runtime | Ollama (Llama3 / Mistral) |
| ML Model | XGBoost / RandomForest (sklearn Pipeline) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | PostgreSQL |
| Logging | structlog (JSON) |
| Deployment | Docker + Docker Compose |

---

## Quick Start — Local Development

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Ollama](https://ollama.com) installed locally (or run via Docker)

### 2. Clone and set up

```bash
git clone <repo-url>
cd "RAG-Decision Engine"

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your PostgreSQL / Ollama settings if needed
```

### 4. Start infrastructure

```bash
# PostgreSQL + MLflow
docker compose up postgres mlflow -d

# Ollama (pull model)
ollama pull llama3
```

### 5. Train the reliability model

```bash
python -m rag_decision_engine.models.train_model
```

### 6. Start the API

```bash
python -m rag_decision_engine.api.server
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 7. Start the dashboard

```bash
streamlit run rag_decision_engine/dashboard/app.py
# → http://localhost:8501
```

---

## Full Docker Deployment

```bash
# Build and start all services
docker compose up --build -d

# Pull the LLM model inside the Ollama container
docker exec rag_ollama ollama pull llama3

# Check health
curl http://localhost:8000/health
```

Services:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

---

## API Reference

### `POST /decision`

Run the full decision pipeline for a query.

**Request**
```json
{
  "query": "Should I use XGBoost or Random Forest for tabular datasets?"
}
```

**Response** — `DecisionReport`
```json
{
  "query": "...",
  "options": [
    {
      "option": "XGBoost",
      "supporting_documents": 5,
      "reliability_score": 0.82,
      "source_credibility": 0.91,
      "final_evidence_score": 0.87,
      "top_evidence_snippets": ["..."]
    }
  ],
  "contradiction_detected": true,
  "contradiction_count": 2,
  "recommended_option": "XGBoost",
  "recommendation_confidence": 0.76,
  "reasoning": "Based on 5 supporting documents ...",
  "latency_ms": 843.2
}
```

### `POST /ingest_documents`

Ingest raw text or file paths into the knowledge base.

```json
{
  "texts": ["XGBoost is a gradient boosting framework..."],
  "file_paths": ["/data/paper.pdf"]
}
```

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

### `GET /metrics`

Operational metrics: request count, decision count, error count, avg latency.

---

## Ingest Documents

### Via API

```bash
curl -X POST http://localhost:8000/ingest_documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["XGBoost is a gradient boosting framework optimised for structured data."]}'
```

### Via Python

```python
from rag_decision_engine.data_pipeline.ingest import IngestionPipeline

pipeline = IngestionPipeline()
pipeline.ingest(["/path/to/paper.pdf", "/path/to/blog.txt"])
```

---

## Train / Retrain the Reliability Model

```bash
# Uses synthetic data by default; swap in real labelled data for production
python -m rag_decision_engine.models.train_model

# Track experiment in MLflow
open http://localhost:5000
```

To use a custom labelled dataset, create a `pandas.DataFrame` with columns:

```
similarity_score, doc_length_tokens, has_citations,
publication_year_norm, source_credibility, chunk_index_norm, label
```

Then call `train(df)` from `rag_decision_engine.models.train_model`.

---

## Evaluation

```bash
python -m rag_decision_engine.evaluation.evaluate
```

Outputs:

```
Retrieval  Recall@10:    0.8400
Retrieval  Precision@10: 0.7200
Retrieval  MRR:          0.6900
Decision   Accuracy:     0.7500
Decision   Avg Confidence: 0.6800
Decision   Contradiction %: 33.33%
Decision   Avg Latency ms: 1240.0
```

---

## Run Tests

```bash
pytest rag_decision_engine/tests/ -v
```

---

## Source Credibility Scale

| Source Type | Credibility Score |
|---|---|
| Research Paper | 0.95 |
| Official Documentation | 0.90 |
| Technical Blog | 0.70 |
| News Article | 0.60 |
| Forum / Stack Overflow | 0.40 |
| Unknown | 0.50 |

Customise in `.env` or `settings.py` via `SOURCE_CREDIBILITY_MAP`.

---

## Evidence Score Formula

```
final_score = 0.5 × similarity_score
            + 0.3 × reliability_score
            + 0.2 × source_credibility
```

Weights are configurable via environment variables:
`WEIGHT_SIMILARITY`, `WEIGHT_RELIABILITY`, `WEIGHT_CREDIBILITY`.

---

## Observability

All components emit structured JSON logs via `structlog`.

Key log fields tracked per request:
- `query_latency_ms`
- `retrieval_time_ms`
- `reranking_time_ms`
- `decision_confidence`
- `contradiction_detected`
- `pipeline_errors`

In production, ship logs to Elasticsearch, Datadog, or Loki.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Reciprocal Rank Fusion for hybrid search | No score normalisation required; robust across score scales |
| XGBoost for reliability model | Superior on tabular feature vectors; fast inference |
| Cross-encoder for reranking + NLI | Better relevance estimation than bi-encoder dot-product |
| Pydantic v2 for all schemas | Fast validation, direct FastAPI serialisation |
| structlog for logging | Structured JSON for machine parsing; human-readable in dev |
| Singleton pattern for models | Avoid reloading large models on every request |

---

## License

MIT
