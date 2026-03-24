import math
from dataclasses import dataclass
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.models.predict import EvidenceFeatures, ReliabilityPredictor
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
logger = get_logger(__name__)
_RECENCY_BASE_YEAR = 2000
_RECENCY_RANGE = 25.0
@dataclass
class ScoredEvidence:
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    reliability_score: float
    source_credibility: float
    citation_score: float
    recency_score: float
    final_score: float
    metadata: dict
class ReliabilityService:
    def __init__(self) -> None:
        self._predictor = ReliabilityPredictor()
        self._credibility_map: dict[str, float] = settings.source_credibility_map
    def score_evidence(
        self,
        documents: list[RetrievedDocument],
    ) -> list[ScoredEvidence]:
        feature_batch = [self._build_features(doc) for doc in documents]
        reliability_scores = self._predictor.score_batch(feature_batch)
        results: list[ScoredEvidence] = []
        for doc, rel_score, features in zip(documents, reliability_scores, feature_batch):
            cred = features.source_credibility
            citation_count = int(doc.metadata.get("citation_count", 0) or 0)
            year = doc.metadata.get("estimated_year") or doc.metadata.get("year")
            cit_score = min(math.log1p(citation_count) / 10.0, 1.0)
            rec_score = max(0.0, min((int(year) - _RECENCY_BASE_YEAR) / _RECENCY_RANGE, 1.0)) if year else 0.0
            final = self._blend(doc.score, rel_score, cred, cit_score, rec_score)
            results.append(
                ScoredEvidence(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    text=doc.text,
                    similarity_score=round(doc.score, 4),
                    reliability_score=round(rel_score, 4),
                    source_credibility=round(cred, 4),
                    citation_score=round(cit_score, 4),
                    recency_score=round(rec_score, 4),
                    final_score=round(final, 4),
                    metadata=doc.metadata,
                )
            )
            logger.debug(
                "evidence_scored",
                chunk_id=doc.chunk_id,
                similarity=round(doc.score, 3),
                reliability=round(rel_score, 3),
                credibility=round(cred, 3),
                citation_score=round(cit_score, 3),
                recency_score=round(rec_score, 3),
                final=round(final, 3),
            )
        results.sort(key=lambda e: e.final_score, reverse=True)
        return results
    def get_source_credibility(self, source_type: str) -> float:
        return self._credibility_map.get(
            source_type,
            self._credibility_map.get("unknown", 0.5),
        )
    def _build_features(self, doc: RetrievedDocument) -> EvidenceFeatures:
        meta = doc.metadata
        origin = str(meta.get("origin", ""))
        source_type = origin if origin in self._credibility_map else str(meta.get("source_type", "unknown"))
        return EvidenceFeatures(
            similarity_score=doc.score,
            doc_length_tokens=int(meta.get("word_count", len(doc.text.split()))),
            has_citations=bool(meta.get("has_citations", False)),
            publication_year=meta.get("estimated_year"),
            source_credibility=self.get_source_credibility(source_type),
            chunk_index=int(meta.get("chunk_index", 0)),
            total_chunks=int(meta.get("total_chunks", 1)),
        )
    @staticmethod
    def _blend(
        similarity: float,
        reliability: float,
        credibility: float,
        citation_score: float = 0.0,
        recency_score: float = 0.0,
    ) -> float:
        return (
            settings.weight_similarity * similarity
            + settings.weight_reliability * reliability
            + settings.weight_credibility * credibility
            + settings.weight_citation * citation_score
            + settings.weight_recency * recency_score
        )
