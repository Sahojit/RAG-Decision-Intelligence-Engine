import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from rag_decision_engine.services.contradiction_service import (
    ContradictionDetector,
    ContradictionReport,
    ContradictionPair,
)
from rag_decision_engine.services.reliability_service import ScoredEvidence


def _make_evidence(chunk_id: str, text: str = "some evidence text") -> ScoredEvidence:
    return ScoredEvidence(
        chunk_id=chunk_id,
        doc_id=chunk_id.rsplit("_", 1)[0],
        text=text,
        similarity_score=0.8,
        reliability_score=0.7,
        source_credibility=0.9,
        citation_score=0.4,
        recency_score=0.5,
        final_score=0.75,
        metadata={},
    )


def _mock_model_high():
    """Return a mock CrossEncoder with logits that softmax to ~0.98 contradiction probability."""
    model = MagicMock()
    logit = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    model.predict.return_value = [logit]
    return model


def _mock_model_low():
    """Return a mock CrossEncoder with logits that softmax to ~0.01 contradiction probability."""
    model = MagicMock()
    logit = np.array([-5.0, 5.0, 5.0], dtype=np.float32)
    model.predict.return_value = [logit]
    return model


class TestContradictionReportDataclass:
    def test_pair_count_empty(self):
        report = ContradictionReport(detected=False, pairs=[], max_contradiction_score=0.0)
        assert report.pair_count == 0

    def test_pair_count_with_pairs(self):
        pair = ContradictionPair("a", "b", "ta", "tb", 0.7)
        report = ContradictionReport(detected=True, pairs=[pair, pair], max_contradiction_score=0.7)
        assert report.pair_count == 2

    def test_detected_false_when_no_pairs(self):
        report = ContradictionReport(detected=False, pairs=[], max_contradiction_score=0.0)
        assert report.detected is False


class TestContradictionDetectorEdgeCases:
    def setup_method(self):
        self.detector = ContradictionDetector(threshold=0.5)

    def test_empty_evidence_returns_no_detection(self):
        report = self.detector.detect([])
        assert report.detected is False
        assert report.pair_count == 0
        assert report.max_contradiction_score == 0.0

    def test_single_evidence_returns_no_detection(self):
        report = self.detector.detect([_make_evidence("doc1_0")])
        assert report.detected is False
        assert report.pair_count == 0


class TestContradictionDetectorWithMockedModel:
    def setup_method(self):
        self.detector = ContradictionDetector(threshold=0.5)

    def test_no_contradiction_below_threshold(self):
        self.detector._model = _mock_model_low()
        evidence = [_make_evidence("doc1_0", "A is good"), _make_evidence("doc2_0", "B is good")]
        report = self.detector.detect(evidence)
        assert report.detected is False
        assert report.pair_count == 0

    def test_contradiction_detected_above_threshold(self):
        self.detector._model = _mock_model_high()
        evidence = [_make_evidence("doc1_0", "A is true"), _make_evidence("doc2_0", "A is false")]
        report = self.detector.detect(evidence)
        assert report.detected is True
        assert report.pair_count >= 1

    def test_max_contradiction_score_populated(self):
        self.detector._model = _mock_model_high()
        evidence = [_make_evidence("doc1_0"), _make_evidence("doc2_0")]
        report = self.detector.detect(evidence)
        assert report.max_contradiction_score > 0.0

    def test_pairs_sorted_by_score_descending(self):
        model = MagicMock()
        high = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        low = np.array([8.0, 0.0, 0.0], dtype=np.float32)
        model.predict.return_value = [high, low, low]
        self.detector._model = model
        self.detector._threshold = 0.5
        evidence = [
            _make_evidence("doc1_0"),
            _make_evidence("doc2_0"),
            _make_evidence("doc3_0"),
        ]
        report = self.detector.detect(evidence)
        if len(report.pairs) > 1:
            scores = [p.contradiction_score for p in report.pairs]
            assert scores == sorted(scores, reverse=True)

    def test_chunk_ids_recorded_in_pair(self):
        self.detector._model = _mock_model_high()
        evidence = [_make_evidence("docA_0"), _make_evidence("docB_0")]
        report = self.detector.detect(evidence)
        if report.pairs:
            ids = {report.pairs[0].chunk_id_a, report.pairs[0].chunk_id_b}
            assert ids == {"docA_0", "docB_0"}


class TestExtractContradictionProb:
    def test_scalar_input(self):
        prob = ContradictionDetector._extract_contradiction_prob(0.7)
        assert abs(prob - 0.7) < 1e-4

    def test_logit_vector_softmax(self):
        logits = np.array([2.0, 0.0, 0.0])
        prob = ContradictionDetector._extract_contradiction_prob(logits)
        expected = np.exp(2.0) / (np.exp(2.0) + 1 + 1)
        assert abs(prob - expected) < 1e-4
