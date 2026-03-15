import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Protocol, Sequence
import sqlalchemy as sa
from sqlalchemy.orm import Session
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.data_pipeline.chunking import DocumentChunk, chunk_document
from rag_decision_engine.data_pipeline.preprocess import clean_text, extract_metadata_hints
logger = get_logger(__name__)
class DocumentLoader(Protocol):
    def load(self, source: str | Path) -> list[tuple[str, str, str]]:
        ...
class PlainTextLoader:
    def load(self, source: str | Path) -> list[tuple[str, str, str]]:
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = _stable_id(str(path))
        return [(doc_id, text, str(path))]
class PDFLoader:
    def load(self, source: str | Path) -> list[tuple[str, str, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf: pip install pypdf") from exc
        path = Path(source)
        reader = PdfReader(str(path))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        doc_id = _stable_id(str(path))
        return [(doc_id, text, str(path))]
class RawTextLoader:
    def load(self, source: str | Path) -> list[tuple[str, str, str]]:
        text = str(source)
        doc_id = _stable_id(text[:200])
        return [(doc_id, text, "raw_input")]
def get_loader(source: str | Path) -> DocumentLoader:
    path = Path(str(source))
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PDFLoader()
    if ext in {".txt", ".md", ".rst"}:
        return PlainTextLoader()
    return RawTextLoader()
class IngestionPipeline:
    def __init__(self) -> None:
        from rag_decision_engine.retrieval.vector_retriever import VectorRetriever
        from rag_decision_engine.services.db import get_engine
        self._vector_retriever = VectorRetriever()
        self._engine = get_engine()
        self._ensure_tables()
    def ingest(
        self,
        sources: Sequence[str | Path],
        force_reingest: bool = False,
    ) -> dict[str, object]:
        t0 = time.perf_counter()
        total_docs = 0
        total_chunks = 0
        skipped = 0
        for source in sources:
            loader = get_loader(source)
            try:
                docs = loader.load(source)
            except Exception as exc:
                logger.error("load_failed", source=str(source), error=str(exc))
                continue
            for doc_id, raw_text, source_path in docs:
                if not force_reingest and self._already_ingested(doc_id):
                    logger.debug("skip_duplicate", doc_id=doc_id)
                    skipped += 1
                    continue
                cleaned = clean_text(raw_text)
                meta_hints = extract_metadata_hints(cleaned, source_path)
                metadata = {"doc_id": doc_id, "source_path": source_path, **meta_hints}
                chunks = chunk_document(doc_id, cleaned, metadata)
                if not chunks:
                    logger.warning("empty_document", doc_id=doc_id)
                    continue
                self._store_chunks(chunks)
                self._record_document(doc_id, source_path, metadata, len(chunks))
                total_docs += 1
                total_chunks += len(chunks)
                logger.info(
                    "document_ingested",
                    doc_id=doc_id,
                    chunks=len(chunks),
                    source_type=meta_hints.get("source_type"),
                )
        elapsed = time.perf_counter() - t0
        summary = {
            "ingested_documents": total_docs,
            "ingested_chunks": total_chunks,
            "skipped_duplicates": skipped,
            "elapsed_seconds": round(elapsed, 3),
        }
        logger.info("ingestion_complete", **summary)
        return summary
    def _store_chunks(self, chunks: list[DocumentChunk]) -> None:
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [{**c.metadata, "text": c.text} for c in chunks]
        self._vector_retriever.add_documents(texts, ids, metadatas)
    def _record_document(
        self,
        doc_id: str,
        source_path: str,
        metadata: dict,
        chunk_count: int,
    ) -> None:
        with Session(self._engine) as session:
            session.execute(
                sa.text(
                    """
                    INSERT INTO documents (doc_id, source_path, metadata, chunk_count)
                    VALUES (:doc_id, :source_path, :metadata, :chunk_count)
                    ON CONFLICT (doc_id) DO UPDATE
                        SET source_path = EXCLUDED.source_path,
                            metadata    = EXCLUDED.metadata,
                            chunk_count = EXCLUDED.chunk_count,
                            updated_at  = NOW()
                    """
                ),
                {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "metadata": json.dumps(metadata),
                    "chunk_count": chunk_count,
                },
            )
            session.commit()
    def _already_ingested(self, doc_id: str) -> bool:
        with Session(self._engine) as session:
            row = session.execute(
                sa.text("SELECT 1 FROM documents WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            ).fetchone()
        return row is not None
    def _ensure_tables(self) -> None:
        with Session(self._engine) as session:
            session.execute(
                sa.text(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id       TEXT PRIMARY KEY,
                        source_path  TEXT,
                        metadata     JSONB,
                        chunk_count  INTEGER DEFAULT 0,
                        created_at   TIMESTAMPTZ DEFAULT NOW(),
                        updated_at   TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
            )
            session.commit()
def _stable_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]
