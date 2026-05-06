import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import configure_logging, get_logger
configure_logging()
logger = get_logger(__name__)
_metrics: dict[str, Any] = defaultdict(lambda: 0)
_metrics["request_count"] = 0
_metrics["decision_count"] = 0
_metrics["ingest_count"] = 0
_metrics["error_count"] = 0
_metrics["total_latency_ms"] = 0.0
_ready = False
_decision_service = None
_ingest_pipeline = None
class DecisionRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    use_live_retrieval: bool = False
class IngestRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    force_reingest: bool = False
def _get_ingest_pipeline():
    global _ingest_pipeline
    if _ingest_pipeline is None:
        from rag_decision_engine.data_pipeline.ingest import IngestionPipeline
        _ingest_pipeline = IngestionPipeline()
    return _ingest_pipeline


def _background_load() -> None:
    """Load DecisionService at startup (FAISS + BM25 only — no PyTorch).
    The embedding model inside VectorRetriever is deferred to the first
    actual encode call, keeping startup RSS well below 512 MB."""
    global _ready, _decision_service
    try:
        from rag_decision_engine.services.decision_service import DecisionService
        _decision_service = DecisionService()
        logger.info("services_ready")
    except Exception as exc:
        logger.error("services_load_failed", error=str(exc))
    finally:
        _ready = True
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", app=settings.app_name, version=settings.app_version)
    t = threading.Thread(target=_background_load, daemon=True)
    t.start()
    yield
    logger.info("shutdown")
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Evidence-based decision intelligence powered by RAG.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.middleware("http")
    async def track_latency(request: Request, call_next):
        t0 = time.perf_counter()
        _metrics["request_count"] += 1
        try:
            response = await call_next(request)
        except Exception:
            _metrics["error_count"] += 1
            raise
        latency = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] += latency
        response.headers["X-Response-Time-Ms"] = str(round(latency, 1))
        return response
    @app.post("/decision", status_code=200)
    async def decision(body: DecisionRequest):
        if not _ready:
            return JSONResponse(status_code=503, content={"detail": "Service loading, retry in a moment."})
        t0 = time.perf_counter()
        logger.info("decision_request", query=body.query)
        try:
            report = _decision_service.decide(body.query, use_live_retrieval=body.use_live_retrieval)
            _metrics["decision_count"] += 1
            _metrics["total_latency_ms"] += (time.perf_counter() - t0) * 1000
            return report
        except Exception as exc:
            _metrics["error_count"] += 1
            logger.error("decision_error", query=body.query, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Decision pipeline failed: {exc}")
    @app.post("/ingest_documents", status_code=202)
    async def ingest_documents(body: IngestRequest):
        if not _ready:
            return JSONResponse(status_code=503, content={"detail": "Service loading, retry in a moment."})
        sources: list[str] = body.texts + body.file_paths
        if not sources:
            raise HTTPException(status_code=422, detail="Provide at least one text or file_path.")
        try:
            summary = _get_ingest_pipeline().ingest(sources, force_reingest=body.force_reingest)
            _metrics["ingest_count"] += summary.get("ingested_documents", 0)
            return {"status": "accepted", **summary}
        except Exception as exc:
            _metrics["error_count"] += 1
            logger.error("ingest_error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name, "version": settings.app_version}
    @app.get("/ready")
    async def ready():
        if not _ready:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {"status": "ready"}
    @app.get("/metrics")
    async def metrics():
        req_count = _metrics["request_count"] or 1
        return {
            "request_count": _metrics["request_count"],
            "decision_count": _metrics["decision_count"],
            "ingest_count": _metrics["ingest_count"],
            "error_count": _metrics["error_count"],
            "avg_latency_ms": round(_metrics["total_latency_ms"] / req_count, 1),
        }
    return app
app = create_app()
if __name__ == "__main__":
    uvicorn.run(
        "rag_decision_engine.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
        reload=False,
    )
