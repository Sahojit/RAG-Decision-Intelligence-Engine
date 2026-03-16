import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import configure_logging, get_logger
from rag_decision_engine.data_pipeline.ingest import IngestionPipeline
from rag_decision_engine.services.decision_service import DecisionReport, DecisionService
configure_logging()
logger = get_logger(__name__)
_metrics: dict[str, Any] = defaultdict(lambda: 0)
_metrics["request_count"] = 0
_metrics["decision_count"] = 0
_metrics["ingest_count"] = 0
_metrics["error_count"] = 0
_metrics["total_latency_ms"] = 0.0
class DecisionRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
class IngestRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    force_reingest: bool = False
_ready = False
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    logger.info("startup", app=settings.app_name, version=settings.app_version)
    loop = asyncio.get_running_loop()
    def _load():
        app.state.decision_service = DecisionService()
        app.state.ingest_pipeline = IngestionPipeline()
    await loop.run_in_executor(None, _load)
    _ready = True
    logger.info("services_ready")
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
    @app.post("/decision", response_model=DecisionReport, status_code=200)
    async def decision(body: DecisionRequest) -> DecisionReport:
        t0 = time.perf_counter()
        logger.info("decision_request", query=body.query)
        try:
            svc: DecisionService = app.state.decision_service
            report = svc.decide(body.query)
            _metrics["decision_count"] += 1
            _metrics["total_latency_ms"] += (time.perf_counter() - t0) * 1000
            return report
        except Exception as exc:
            _metrics["error_count"] += 1
            logger.error("decision_error", query=body.query, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Decision pipeline failed: {exc}")
    @app.post("/ingest_documents", status_code=202)
    async def ingest_documents(body: IngestRequest) -> dict:
        sources: list[str] = body.texts + body.file_paths
        if not sources:
            raise HTTPException(status_code=422, detail="Provide at least one text or file_path.")
        try:
            pipeline: IngestionPipeline = app.state.ingest_pipeline
            summary = pipeline.ingest(sources, force_reingest=body.force_reingest)
            _metrics["ingest_count"] += summary.get("ingested_documents", 0)
            return {"status": "accepted", **summary}
        except Exception as exc:
            _metrics["error_count"] += 1
            logger.error("ingest_error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    @app.get("/health")
    async def health() -> dict:
        if not _ready:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }
    @app.get("/metrics")
    async def metrics() -> dict:
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
        workers=settings.api_workers,
        reload=settings.environment == "development",
    )
