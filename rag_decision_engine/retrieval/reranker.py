import time
from typing import Optional
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
logger = get_logger(__name__)
class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._model: Optional[object] = None
    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        k = top_k or settings.rerank_top_k
        if not documents:
            return []
        model = self._get_model()
        t0 = time.perf_counter()
        import numpy as np
        pairs = [(query, doc.text) for doc in documents]
        raw_scores = model.predict(pairs, show_progress_bar=False)
        scores = [float(1.0 / (1.0 + np.exp(-s))) for s in raw_scores]
        reranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        results = []
        for doc, score in reranked:
            results.append(
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata,
                    retriever="reranked",
                )
            )
        logger.info(
            "reranking_complete",
            query_preview=query[:60],
            candidates=len(documents),
            returned=len(results),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return results
    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                ) from exc
            logger.info("loading_reranker", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model
