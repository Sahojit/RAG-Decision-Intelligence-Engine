import time
from collections import defaultdict
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.bm25_retriever import BM25Retriever
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument, VectorRetriever
logger = get_logger(__name__)
_RRF_K = 60
class HybridRetriever:
    def __init__(
        self,
        vector_weight: float | None = None,
    ) -> None:
        self._vector = VectorRetriever()
        self._bm25 = BM25Retriever()
        self._vector_weight = vector_weight or settings.vector_weight
        self._bm25_weight = 1.0 - self._vector_weight
        self._seed_bm25_from_faiss()
    def _seed_bm25_from_faiss(self) -> None:
        meta_list = self._vector._metadata
        if not meta_list:
            return
        texts, ids, metas = [], [], []
        for m in meta_list:
            text = m.get("text", "")
            if text:
                texts.append(text)
                ids.append(m.get("chunk_id", ""))
                metas.append(m)
        if texts:
            self._bm25.add_documents(texts, ids, metas)
            logger.info("bm25_seeded_from_faiss", docs=len(texts))
    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict],
        texts_with_content: list[str] | None = None,
    ) -> None:
        self._vector.add_documents(texts, ids, metadatas)
        full_texts = texts_with_content or texts
        self._bm25.add_documents(full_texts, ids, metadatas)
    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedDocument]:
        k = top_k or settings.rerank_top_k
        c_k = candidate_k or settings.retrieval_top_k
        t0 = time.perf_counter()
        vector_results = self._vector.search(query, top_k=c_k)
        bm25_results = self._bm25.search(query, top_k=c_k)
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k=k)
        logger.info(
            "hybrid_search",
            query_preview=query[:60],
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            fused_results=len(fused),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return fused
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievedDocument],
        bm25_results: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, RetrievedDocument] = {}
        for rank, doc in enumerate(vector_results, start=1):
            scores[doc.chunk_id] += self._vector_weight / (_RRF_K + rank)
            doc_map[doc.chunk_id] = doc
        for rank, doc in enumerate(bm25_results, start=1):
            scores[doc.chunk_id] += self._bm25_weight / (_RRF_K + rank)
            if doc.chunk_id not in doc_map:
                doc_map[doc.chunk_id] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, fused_score in ranked:
            doc = doc_map[chunk_id]
            results.append(
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=fused_score,
                    metadata=doc.metadata,
                    retriever="hybrid",
                )
            )
        return results
