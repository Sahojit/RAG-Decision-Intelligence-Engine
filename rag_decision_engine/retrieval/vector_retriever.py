import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import faiss
import numpy as np
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# Singleton — loaded once on first encode call, never at import or __init__ time.
# Keeping the import lazy (inside _get_embedding_model) means importing this module
# does NOT pull in torch, which would cost ~150 MB and OOM the 512 MB free instance
# before any request is ever handled.
_EMBEDDING_MODEL: Optional["SentenceTransformer"] = None


def _get_embedding_model() -> "SentenceTransformer":
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy — defers torch import
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _EMBEDDING_MODEL = SentenceTransformer(settings.embedding_model, device="cpu")
    return _EMBEDDING_MODEL


@dataclass
class RetrievedDocument:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)
    retriever: str = "unknown"
class VectorRetriever:
    def __init__(self) -> None:
        # Embedding model is NOT loaded here — deferred to first _embed() call.
        # Loading torch + weights at __init__ time causes OOM on 512 MB hosts because
        # both DecisionService and IngestionPipeline create a VectorRetriever at startup.
        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []
        self._load_or_create_index()
    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict],
    ) -> None:
        if not texts:
            return
        t0 = time.perf_counter()
        embeddings = self._embed(texts)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        for chunk_id, meta in zip(ids, metadatas):
            self._metadata.append({"chunk_id": chunk_id, **meta})
        self._persist()
        logger.debug(
            "vectors_added",
            count=len(texts),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        k = top_k or settings.retrieval_top_k
        if self._index is None or self._index.ntotal == 0:
            logger.warning("vector_index_empty")
            return []
        t0 = time.perf_counter()
        q_vec = self._embed([query])
        faiss.normalize_L2(q_vec)
        actual_k = min(k, self._index.ntotal)
        distances, indices = self._index.search(q_vec, actual_k)
        results: list[RetrievedDocument] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedDocument(
                    chunk_id=meta.get("chunk_id", str(idx)),
                    doc_id=meta.get("doc_id", ""),
                    text=meta.get("text", ""),
                    score=float(dist),
                    metadata=meta,
                    retriever="vector",
                )
            )
        logger.debug(
            "vector_search",
            query_preview=query[:60],
            results=len(results),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return results
    @property
    def document_count(self) -> int:
        return self._index.ntotal if self._index else 0
    def _embed(self, texts: list[str]) -> np.ndarray:
        return _get_embedding_model().encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        ).astype(np.float32)
    def _load_or_create_index(self) -> None:
        index_path = settings.faiss_index_path
        meta_path = settings.faiss_metadata_path
        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._metadata = json.loads(meta_path.read_text())
            logger.info(
                "faiss_index_loaded",
                vectors=self._index.ntotal,
                path=str(index_path),
            )
        else:
            self._index = faiss.IndexFlatIP(settings.embedding_dim)
            self._metadata = []
            logger.info("faiss_index_created", dim=settings.embedding_dim)
    def _persist(self) -> None:
        index_path = settings.faiss_index_path
        meta_path = settings.faiss_metadata_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(json.dumps(self._metadata, default=str))
