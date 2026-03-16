# CLAUDE.md — RAG Decision Intelligence Engine

## Project
Production-grade evidence retrieval and decision analysis system.
Python 3.11, FastAPI, Streamlit, FAISS, BM25, XGBoost, Ollama (Llama3), PostgreSQL, MLflow, Docker.

## Working Directory
`/Users/sahojitkarmakar/Documents/project/Rag Projects/ RAG-Decision Engine`

## Architecture
```
User Query → Decision Parser → Hybrid Retrieval (FAISS+BM25+RRF) →
CrossEncoder Reranker → Reliability Model (XGBoost) →
Credibility Scoring → Contradiction Detection (NLI) →
Decision Policy Engine → Self-Reflection Agent →
LLM Reasoning (Ollama/Llama3) → DecisionReport (Pydantic)
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
- Do not commit `.env` — it contains local credentials

### Always do this
- Use `from langchain_text_splitters import RecursiveCharacterTextSplitter` (not `langchain.text_splitter`)
- Use `from langchain_core.prompts import PromptTemplate`
- Use LCEL for chains: `chain = prompt | llm; result = chain.invoke({...})`
- Store chunk `text` in FAISS metadata: `metadatas = [{**c.metadata, "text": c.text} for c in chunks]`
- Apply sigmoid to CrossEncoder raw logits: `score = 1.0 / (1.0 + exp(-logit))`
- Seed BM25 from FAISS metadata on startup (BM25 is in-memory, lost on restart)
- Load ML models in a daemon thread at startup so `/health` responds immediately
- `/health` always returns 200 (liveness). `/ready` returns 503 while models load, 200 when ready

## Evidence Scoring Formula
```
final_score = 0.5 × similarity_score
            + 0.3 × reliability_score
            + 0.2 × source_credibility
```
Configurable via `WEIGHT_SIMILARITY`, `WEIGHT_RELIABILITY`, `WEIGHT_CREDIBILITY` env vars.

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

## Railway Deployment
- `railway.toml` configured at repo root
- Healthcheck path: `/health` (always 200)
- Models download at runtime via daemon thread on first boot
- Do NOT add model pre-download to Dockerfile — Railway build containers OOM with PyTorch
- Do NOT use GPU torch — pin `torch==2.2.2+cpu` with `--extra-index-url https://download.pytorch.org/whl/cpu`

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

## Environment Variables (key ones)
| Var | Default | Notes |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
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
| `data_pipeline/ingest.py` | End-to-end ingestion |
| `retrieval/hybrid_retriever.py` | RRF fusion + BM25 seeding |
| `retrieval/reranker.py` | CrossEncoder + sigmoid norm |
| `models/train_model.py` | XGBoost training + MLflow |
| `models/predict.py` | ReliabilityPredictor singleton |
| `services/decision_service.py` | Full pipeline orchestrator |
| `services/decision_policy.py` | Policy engine (contradiction/margin rules) |
| `services/reflection_service.py` | Post-decision evidence audit |
| `services/contradiction_service.py` | NLI pairwise detection |
| `api/server.py` | FastAPI app |
| `dashboard/app.py` | Streamlit UI |
