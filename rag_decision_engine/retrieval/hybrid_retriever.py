import time
from collections import defaultdict
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.bm25_retriever import BM25Retriever
from rag_decision_engine.retrieval.metadata_filter import apply_metadata_filter
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument, VectorRetriever
from rag_decision_engine.services.live_retrieval import detect_query_type
from rag_decision_engine.services.query_filter_parser import QueryFilters
logger = get_logger(__name__)
_RRF_K = 60
_RESEARCH_VECTOR_WEIGHT = 0.8
_RESEARCH_BM25_WEIGHT = 0.2
class HybridRetriever:
    def __init__(self, vector_weight: float | None = None) -> None:
        self._vector = VectorRetriever()
        self._bm25 = BM25Retriever()
        self._vector_weight = vector_weight or settings.vector_weight
        self._bm25_weight = 1.0 - self._vector_weight
        self._seed_bm25_from_faiss()
    def _seed_bm25_from_faiss(self) -> None:
        texts, ids, metas = [], [], []
        for m in self._vector._metadata:
            text = m.get("text", "")
            if text:
                texts.append(text)
                ids.append(m.get("chunk_id", ""))
                metas.append(m)
        if texts:
            self._bm25.add_documents(texts, ids, metas)
            logger.info("bm25_seeded", docs=len(texts))
    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict],
        texts_with_content: list[str] | None = None,
    ) -> None:
        self._vector.add_documents(texts, ids, metadatas)
        self._bm25.add_documents(texts_with_content or texts, ids, metadatas)
    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: QueryFilters | None = None,
        dynamic_docs: list[RetrievedDocument] | None = None,
    ) -> list[RetrievedDocument]:
        k = top_k or settings.rerank_top_k
        c_k = candidate_k or settings.retrieval_top_k
        if detect_query_type(query):
            v_weight = _RESEARCH_VECTOR_WEIGHT
            b_weight = _RESEARCH_BM25_WEIGHT
        else:
            v_weight = self._vector_weight
            b_weight = self._bm25_weight
        t0 = time.perf_counter()
        vector_results = self._vector.search(query, top_k=c_k)
        bm25_results = self._bm25.search(query, top_k=c_k)
        fused = self._rrf(vector_results, bm25_results, dynamic_docs or [], top_k=k, vector_weight=v_weight, bm25_weight=b_weight)
        if filters and not filters.is_empty():
            fused = apply_metadata_filter(fused, filters)
        if dynamic_docs:
            logger.info("dynamic_docs_merged", count=len(dynamic_docs))
        logger.info(
            "hybrid_search",
            query_preview=query[:60],
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            dynamic_docs=len(dynamic_docs or []),
            fused=len(fused),
            vector_weight=v_weight,
            bm25_weight=b_weight,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return fused
    def _rrf(
        self,
        vector_results: list[RetrievedDocument],
        bm25_results: list[RetrievedDocument],
        dynamic_docs: list[RetrievedDocument],
        top_k: int,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> list[RetrievedDocument]:
        vw = vector_weight if vector_weight is not None else self._vector_weight
        bw = bm25_weight if bm25_weight is not None else self._bm25_weight
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, RetrievedDocument] = {}
        for rank, doc in enumerate(vector_results, start=1):
            scores[doc.chunk_id] += vw / (_RRF_K + rank)
            doc_map[doc.chunk_id] = doc
        for rank, doc in enumerate(bm25_results, start=1):
            scores[doc.chunk_id] += bw / (_RRF_K + rank)
            if doc.chunk_id not in doc_map:
                doc_map[doc.chunk_id] = doc
        for rank, doc in enumerate(dynamic_docs, start=1):
            if doc.chunk_id not in doc_map:
                scores[doc.chunk_id] += 0.3 / (_RRF_K + rank)
                doc_map[doc.chunk_id] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedDocument(
                chunk_id=doc_map[cid].chunk_id,
                doc_id=doc_map[cid].doc_id,
                text=doc_map[cid].text,
                score=score,
                metadata=doc_map[cid].metadata,
                retriever="hybrid",
            )
            for cid, score in ranked
        ]
