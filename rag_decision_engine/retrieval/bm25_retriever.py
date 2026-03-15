import re
import time
from typing import Optional
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
logger = get_logger(__name__)
def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
class BM25Retriever:
    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._documents: list[dict] = []
        self._bm25: Optional[object] = None
    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict],
    ) -> None:
        for text, chunk_id, meta in zip(texts, ids, metadatas):
            self._corpus_tokens.append(_tokenise(text))
            self._documents.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id", ""),
                    "text": text,
                    "metadata": meta,
                }
            )
        self._rebuild_index()
        logger.debug("bm25_corpus_updated", total_docs=len(self._documents))
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        k = top_k or settings.retrieval_top_k
        if self._bm25 is None or not self._documents:
            logger.warning("bm25_corpus_empty")
            return []
        t0 = time.perf_counter()
        query_tokens = _tokenise(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        results = []
        max_score = ranked[0][1] if ranked else 1.0
        for idx, raw_score in ranked:
            if raw_score <= 0:
                break
            doc = self._documents[idx]
            results.append(
                RetrievedDocument(
                    chunk_id=doc["chunk_id"],
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    score=float(raw_score / max_score) if max_score > 0 else 0.0,
                    metadata=doc["metadata"],
                    retriever="bm25",
                )
            )
        logger.debug(
            "bm25_search",
            query_preview=query[:60],
            results=len(results),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return results
    @property
    def corpus_size(self) -> int:
        return len(self._documents)
    def _rebuild_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank_bm25: pip install rank-bm25") from exc
        self._bm25 = BM25Okapi(self._corpus_tokens)
