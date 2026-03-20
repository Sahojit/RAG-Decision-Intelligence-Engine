# CLAUDE.md — Evidentia: AI Tech Decision Engine

## Project
Production-grade tech comparison assistant. Engineers ask "X vs Y?" and the system retrieves evidence, scores reliability with ML, detects contradictions via NLI, and returns a structured decision report.
Python 3.11, FastAPI, Streamlit, FAISS, BM25, XGBoost, Ollama (Llama3), PostgreSQL, MLflow, Docker.

## Working Directory
`/Users/sahojitkarmakar/Documents/project/Rag Projects/ RAG-Decision Engine`

## Architecture
```
User Query
↓
Query Filter Parser (source_type, year, citations)
↓
Query Type Detection (ML/research → trigger live retrieval)
↓
Live Retrieval (arXiv + Semantic Scholar) [optional]
↓
Hybrid Retrieval (FAISS + BM25 + RRF + dynamic_docs)
↓
Metadata Filtering (pre-rerank, with fallback if <5 results)
↓
CrossEncoder Reranker (sigmoid-normalized)
↓
Reliability Scoring (XGBoost, 6 features)
↓
Contradiction Detection (NLI, DeBERTa)
↓
Decision Engine (strong / weak / inconclusive)
↓
Confidence Validator (penalty-based adjustment)
↓
LLM Reasoning (Ollama/Llama3 via LCEL, fallback if unavailable)
↓
DecisionReport (Pydantic)
```

## Key Rules

### Never do this
- Do not add `from __future__ import annotations` to `api/server.py` — breaks FastAPI type resolution
- Do not define Pydantic models inside factory functions — FastAPI cannot resolve them
- Do not use `langchain_classic` — not a real PyPI package
- Do not use `LLMChain` from langchain — removed in 0.3+, use LCEL: `chain = prompt | llm`
- Do not run uvicorn with `reload=True` in production — triggers infinite restart loop
- Do not use `structlog.stdlib.add_logger_name` — incompatible with `PrintLoggerFactory`
- Do not add `use_label_encoder=False` to XGBClassifier — deprecated in XGBoost v2+
- Do not load ML models on the request path — always use singletons loaded at startup
- Do not add model pre-download to Dockerfile — Railway/CI build containers OOM with PyTorch
- Do not pin `torch==2.2.2+cpu` in requirements.txt — breaks arm64 (Apple Silicon) Docker builds
- Do not commit `.env` — it contains local credentials
- Do not import `decision_policy` or `reflection_service` — both deleted, replaced by `decision_engine` and `confidence_validator`

### Always do this
- Use `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Use `from langchain_core.prompts import PromptTemplate`
- Use LCEL for chains: `chain = prompt | llm; result = chain.invoke({...})`
- Store chunk `text` in FAISS metadata: `metadatas = [{**c.metadata, "text": c.text} for c in chunks]`
- Apply sigmoid to CrossEncoder raw logits: `score = 1.0 / (1.0 + exp(-logit))`
- Seed BM25 from FAISS metadata on startup (BM25 is in-memory, lost on restart)
- Load ML models in a daemon thread at startup so `/health` responds immediately
- `/health` always returns 200 (liveness). `/ready` returns 503 while loading, 200 when ready
- Pass `filters` and `dynamic_docs` to `HybridRetriever.search()` — not applied elsewhere
- Use `_QUERY_CACHE` dict in `DecisionService` for query-level caching (max 100 entries)

## Decision Engine Logic
```
IF contradiction_score > 0.6         → inconclusive
ELIF score_margin < 0.03             → inconclusive (too close)
ELIF score_margin < 0.10             → weak winner
ELSE                                 → strong winner
```
Returns: `recommendation`, `decision_type`, `confidence`, `key_factors`

## Confidence Validator Penalties
```
sparse evidence (<3 chunks)          → -0.12
low avg credibility (<0.60)          → -0.08
few high-quality sources (<30%)      → -0.07
low avg reliability (<0.45)          → -0.06
single document concentration        → -0.10
```

## Evidence Scoring Formula
```
final_score = 0.5 × similarity_score
            + 0.3 × reliability_score
            + 0.2 × source_credibility
```
Configurable via `WEIGHT_SIMILARITY`, `WEIGHT_RELIABILITY`, `WEIGHT_CREDIBILITY` env vars.

## Source Credibility Map
```
research_paper → 0.95
documentation  → 0.90
blog           → 0.70
forum          → 0.40
unknown        → 0.50
```

## DecisionReport Schema
```python
query: str
options: list[OptionResult]          # name, evidence_score, reliability, credibility, supporting_docs
recommendation: Optional[str]
decision_type: str                   # "strong" | "weak" | "inconclusive"
confidence: float
key_factors: list[str]
contradictions: int
sources_summary: SourcesSummary      # research_papers, documentation, blogs, forums, live_api
reasoning: str
latency_ms: float
live_retrieval_used: bool
live_docs_count: int
local_docs_count: int
filters_applied: dict
validation_warning: bool
validation_reason: str
```

## Docker
```bash
docker compose up --build -d
docker exec rag_ollama ollama pull llama3
curl http://localhost:8080/health
```
- API: http://localhost:8080
- Dashboard: http://localhost:8090
- MLflow: http://localhost:6001
- PostgreSQL: localhost:5440
- Ollama: localhost:11435

## Railway Deployment
- `railway.toml` configured at repo root
- Healthcheck path: `/health` (always 200)
- Models download at runtime via daemon thread on first boot
- Do NOT add model pre-download to Dockerfile — build containers OOM
- Do NOT use GPU torch — Railway runs x86_64 Linux, CPU torch installs automatically

## Local Dev Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env
docker compose up postgres mlflow -d
ollama pull llama3
python -m rag_decision_engine.models.train_model
python -m rag_decision_engine.api.server
streamlit run rag_decision_engine/dashboard/app.py
```

## Docker Hub Images
- `sahojit/rag-api:latest`
- `sahojit/rag-dashboard:latest`
- `sahojit/rag-ollama:latest`

## GitHub
https://github.com/Sahojit/AIOps-Pipeline-RCA

## Environment Variables
| Var | Default | Notes |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | Set to `postgres` inside Docker |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://host.docker.internal:11434` in Docker |
| `OLLAMA_MODEL` | `llama3` | |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | |
| `RETRIEVAL_TOP_K` | `20` | candidates before reranking |
| `RERANK_TOP_K` | `10` | top-k after reranking |
| `CONTRADICTION_THRESHOLD` | `0.5` | NLI threshold |

## Module Quick Reference
| File | Purpose |
|---|---|
| `config/settings.py` | All env config (pydantic-settings) |
| `config/logging_config.py` | Structlog JSON setup |
| `data_pipeline/ingest.py` | End-to-end ingestion into FAISS + PostgreSQL |
| `retrieval/hybrid_retriever.py` | RRF fusion — FAISS + BM25 + dynamic_docs, filters post-merge |
| `retrieval/metadata_filter.py` | Pre-rerank filter on source_type, year, citations |
| `retrieval/reranker.py` | CrossEncoder + sigmoid normalization |
| `models/train_model.py` | XGBoost training + MLflow tracking |
| `models/predict.py` | ReliabilityPredictor singleton, batch scoring |
| `services/query_filter_parser.py` | Extract structured filters from natural language |
| `services/live_retrieval.py` | Async arXiv + Semantic Scholar fetch |
| `services/reliability_service.py` | Evidence scoring (similarity + reliability + credibility) |
| `services/contradiction_service.py` | NLI pairwise contradiction detection (DeBERTa) |
| `services/decision_engine.py` | strong/weak/inconclusive logic — replaces decision_policy.py |
| `services/confidence_validator.py` | Penalty-based confidence adjustment — replaces reflection_service.py |
| `services/decision_service.py` | Full pipeline orchestrator + query cache |
| `api/server.py` | FastAPI — /health /ready /decision /ingest_documents /metrics |
| `dashboard/app.py` | Streamlit — verdict card, quality badges, collapsible debug |
| `evaluation/evaluate.py` | Recall@k, MRR, accuracy, contradiction rate |
