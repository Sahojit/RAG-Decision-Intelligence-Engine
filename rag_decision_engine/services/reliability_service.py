from dataclasses import dataclass
from typing import Optional
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.models.predict import EvidenceFeatures, ReliabilityPredictor
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
logger = get_logger(__name__)
@dataclass
class ScoredEvidence:
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    reliability_score: float
    source_credibility: float
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
            final = self._blend(doc.score, rel_score, cred)
            results.append(
                ScoredEvidence(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    text=doc.text,
                    similarity_score=round(doc.score, 4),
                    reliability_score=round(rel_score, 4),
                    source_credibility=round(cred, 4),
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
        source_type = str(meta.get("source_type", "unknown"))
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
    def _blend(similarity: float, reliability: float, credibility: float) -> float:
        return (
            settings.weight_similarity * similarity
            + settings.weight_reliability * reliability
            + settings.weight_credibility * credibility
        )
