import pytest
from rag_decision_engine.models.predict import EvidenceFeatures, ReliabilityPredictor
from rag_decision_engine.services.reliability_service import ReliabilityService
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
class TestEvidenceFeatures:
    def test_year_normalisation_2025(self):
        f = EvidenceFeatures(0.8, 500, True, 2025, 0.9, 0, 10)
        arr = f.to_array()
        assert arr[3] == pytest.approx(1.0, abs=0.01)
    def test_year_normalisation_2000(self):
        f = EvidenceFeatures(0.8, 500, True, 2000, 0.9, 0, 10)
        arr = f.to_array()
        assert arr[3] == pytest.approx(0.0, abs=0.01)
    def test_year_none_returns_half(self):
        f = EvidenceFeatures(0.8, 500, True, None, 0.9, 0, 10)
        arr = f.to_array()
        assert arr[3] == pytest.approx(0.5, abs=0.01)
    def test_array_length(self):
        f = EvidenceFeatures(0.8, 500, True, 2022, 0.9, 0, 10)
        assert len(f.to_array()) == 6
class TestReliabilityPredictor:
    def test_score_returns_float_in_range(self):
        pred = ReliabilityPredictor()
        pred._pipeline = None
        features = EvidenceFeatures(
            similarity_score=0.8,
            doc_length_tokens=400,
            has_citations=True,
            publication_year=2023,
            source_credibility=0.9,
            chunk_index=0,
            total_chunks=5,
        )
        score = pred.score(features)
        assert 0.0 <= score <= 1.0
    def test_high_quality_scores_higher(self):
        pred = ReliabilityPredictor()
        pred._pipeline = None
        high = EvidenceFeatures(0.9, 800, True, 2024, 0.95, 0, 5)
        low = EvidenceFeatures(0.2, 50, False, 2000, 0.4, 0, 5)
        assert pred.score(high) > pred.score(low)
class TestReliabilityService:
    def _make_doc(self, chunk_id: str, score: float, source_type: str = "research_paper"):
        return RetrievedDocument(
            chunk_id=chunk_id,
            doc_id="d1",
            text="This is a test document about machine learning techniques.",
            score=score,
            metadata={
                "source_type": source_type,
                "has_citations": True,
                "estimated_year": 2023,
                "word_count": 100,
                "chunk_index": 0,
                "total_chunks": 3,
            },
            retriever="hybrid",
        )
    def test_scores_list(self):
        svc = ReliabilityService()
        docs = [self._make_doc(f"c{i}", 0.7) for i in range(3)]
        scored = svc.score_evidence(docs)
        assert len(scored) == 3
        for s in scored:
            assert 0.0 <= s.final_score <= 1.0
    def test_sorted_by_final_score(self):
        svc = ReliabilityService()
        docs = [
            self._make_doc("c1", 0.9),
            self._make_doc("c2", 0.3),
            self._make_doc("c3", 0.6),
        ]
        scored = svc.score_evidence(docs)
        scores = [s.final_score for s in scored]
        assert scores == sorted(scores, reverse=True)
    def test_get_source_credibility_known(self):
        svc = ReliabilityService()
        assert svc.get_source_credibility("research_paper") >= 0.9
    def test_get_source_credibility_unknown(self):
        svc = ReliabilityService()
        score = svc.get_source_credibility("alien_source")
        assert 0.0 <= score <= 1.0
