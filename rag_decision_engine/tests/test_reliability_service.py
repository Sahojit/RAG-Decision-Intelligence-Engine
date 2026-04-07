import pytest
from rag_decision_engine.services.reliability_service import ReliabilityService, ScoredEvidence
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
from rag_decision_engine.models.predict import EvidenceFeatures, ReliabilityPredictor


def _make_doc(
    chunk_id: str = "doc1_0",
    score: float = 0.8,
    metadata: dict | None = None,
    text: str = "evidence text sample",
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        doc_id=chunk_id.rsplit("_", 1)[0],
        text=text,
        score=score,
        metadata=metadata or {},
        retriever="vector",
    )


class TestReliabilityPredictor:
    def setup_method(self):
        ReliabilityPredictor._instance = None

    def test_heuristic_score_range(self):
        predictor = ReliabilityPredictor()
        features = EvidenceFeatures(
            similarity_score=0.8,
            doc_length_tokens=200,
            has_citations=True,
            publication_year=2022,
            source_credibility=0.9,
            chunk_index=0,
            total_chunks=3,
        )
        score = predictor.score(features)
        assert 0.0 <= score <= 1.0

    def test_heuristic_higher_credibility_gives_higher_score(self):
        predictor = ReliabilityPredictor()
        low = EvidenceFeatures(0.8, 200, False, 2020, 0.2, 0)
        high = EvidenceFeatures(0.8, 200, False, 2020, 0.9, 0)
        assert predictor.score(high) > predictor.score(low)

    def test_heuristic_citations_boost_score(self):
        predictor = ReliabilityPredictor()
        no_cite = EvidenceFeatures(0.8, 200, False, 2020, 0.7, 0)
        with_cite = EvidenceFeatures(0.8, 200, True, 2020, 0.7, 0)
        assert predictor.score(with_cite) > predictor.score(no_cite)

    def test_batch_score_matches_individual(self):
        predictor = ReliabilityPredictor()
        features = [
            EvidenceFeatures(0.7, 150, True, 2021, 0.8, 0),
            EvidenceFeatures(0.5, 300, False, 2019, 0.6, 1),
        ]
        batch = predictor.score_batch(features)
        individual = [predictor.score(f) for f in features]
        for b, i in zip(batch, individual):
            assert abs(b - i) < 1e-5

    def test_singleton_returns_same_instance(self):
        p1 = ReliabilityPredictor()
        p2 = ReliabilityPredictor()
        assert p1 is p2


class TestEvidenceFeaturesToArray:
    def test_array_length(self):
        f = EvidenceFeatures(0.8, 200, True, 2022, 0.9, 2, total_chunks=5)
        arr = f.to_array()
        assert arr.shape == (6,)

    def test_chunk_index_normalised(self):
        f = EvidenceFeatures(0.8, 200, True, 2022, 0.9, chunk_index=2, total_chunks=4)
        arr = f.to_array()
        assert abs(arr[-1] - 0.5) < 1e-5

    def test_none_year_returns_half(self):
        f = EvidenceFeatures(0.8, 200, False, None, 0.7, 0)
        arr = f.to_array()
        assert abs(arr[3] - 0.5) < 1e-5

    def test_year_normalisation_clamps_to_zero_one(self):
        f_old = EvidenceFeatures(0.8, 200, False, 1990, 0.7, 0)
        f_new = EvidenceFeatures(0.8, 200, False, 2030, 0.7, 0)
        assert f_old.to_array()[3] == 0.0
        assert f_new.to_array()[3] == 1.0


class TestReliabilityServiceScoring:
    def setup_method(self):
        ReliabilityPredictor._instance = None
        self.service = ReliabilityService()

    def test_score_evidence_returns_scored_list(self):
        docs = [_make_doc(chunk_id=f"doc{i}_0") for i in range(3)]
        results = self.service.score_evidence(docs)
        assert len(results) == 3
        assert all(isinstance(r, ScoredEvidence) for r in results)

    def test_scored_evidence_sorted_by_final_score(self):
        docs = [
            _make_doc("doc1_0", score=0.9),
            _make_doc("doc2_0", score=0.3),
            _make_doc("doc3_0", score=0.6),
        ]
        results = self.service.score_evidence(docs)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_credibility_map_lookup_known_source(self):
        cred = self.service.get_source_credibility("arxiv")
        assert cred == 0.90

    def test_credibility_map_lookup_unknown_source(self):
        cred = self.service.get_source_credibility("some_random_source")
        assert 0.0 < cred <= 1.0

    def test_metadata_origin_used_for_credibility(self):
        docs = [_make_doc("doc1_0", metadata={"origin": "arxiv", "word_count": 200})]
        results = self.service.score_evidence(docs)
        assert results[0].source_credibility == 0.90

    def test_citation_count_affects_citation_score(self):
        no_cite = _make_doc("doc1_0", metadata={"citation_count": 0})
        high_cite = _make_doc("doc2_0", metadata={"citation_count": 1000})
        results = self.service.score_evidence([no_cite, high_cite])
        cite_map = {r.chunk_id: r.citation_score for r in results}
        assert cite_map["doc2_0"] > cite_map["doc1_0"]

    def test_recency_score_newer_higher(self):
        old = _make_doc("doc1_0", metadata={"estimated_year": 2005})
        new = _make_doc("doc2_0", metadata={"estimated_year": 2023})
        results = self.service.score_evidence([old, new])
        rec_map = {r.chunk_id: r.recency_score for r in results}
        assert rec_map["doc2_0"] > rec_map["doc1_0"]

    def test_final_score_in_valid_range(self):
        docs = [_make_doc(f"doc{i}_0") for i in range(5)]
        results = self.service.score_evidence(docs)
        for r in results:
            assert 0.0 <= r.final_score <= 1.5
