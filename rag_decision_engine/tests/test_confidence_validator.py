import pytest
from rag_decision_engine.services.confidence_validator import ConfidenceValidator, ValidationResult
from rag_decision_engine.services.reliability_service import ScoredEvidence


def _make_evidence(
    chunk_id: str = "doc1_0",
    credibility: float = 0.85,
    reliability: float = 0.75,
) -> ScoredEvidence:
    return ScoredEvidence(
        chunk_id=chunk_id,
        doc_id=chunk_id.rsplit("_", 1)[0],
        text="sample evidence text",
        similarity_score=0.8,
        reliability_score=reliability,
        source_credibility=credibility,
        citation_score=0.5,
        recency_score=0.6,
        final_score=0.75,
        metadata={},
    )


def _good_evidence(n: int = 5) -> list[ScoredEvidence]:
    return [
        _make_evidence(chunk_id=f"doc{i}_0", credibility=0.85, reliability=0.75)
        for i in range(n)
    ]


class TestConfidenceValidatorNoEvidence:
    def setup_method(self):
        self.validator = ConfidenceValidator()

    def test_empty_evidence_returns_zero_confidence(self):
        result = self.validator.validate([], 0.9)
        assert result.adjusted_confidence == 0.0

    def test_empty_evidence_sets_warning_flag(self):
        result = self.validator.validate([], 0.9)
        assert result.warning_flag is True

    def test_empty_evidence_reason_mentions_no_evidence(self):
        result = self.validator.validate([], 0.9)
        assert "no evidence" in result.validation_reason.lower()


class TestConfidenceValidatorGoodEvidence:
    def setup_method(self):
        self.validator = ConfidenceValidator()

    def test_good_evidence_no_penalty(self):
        evidence = _good_evidence(5)
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence == 0.80

    def test_good_evidence_no_warning(self):
        evidence = _good_evidence(5)
        result = self.validator.validate(evidence, 0.80)
        assert result.warning_flag is False

    def test_returns_validation_result_instance(self):
        result = self.validator.validate(_good_evidence(), 0.8)
        assert isinstance(result, ValidationResult)


class TestConfidenceValidatorPenalties:
    def setup_method(self):
        self.validator = ConfidenceValidator()

    def test_sparse_evidence_reduces_confidence(self):
        evidence = [_make_evidence(chunk_id=f"doc{i}_0") for i in range(2)]
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence < 0.80
        assert result.warning_flag is True

    def test_low_credibility_reduces_confidence(self):
        evidence = [
            _make_evidence(chunk_id=f"doc{i}_0", credibility=0.30, reliability=0.75)
            for i in range(5)
        ]
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence < 0.80
        assert result.warning_flag is True

    def test_low_reliability_reduces_confidence(self):
        evidence = [
            _make_evidence(chunk_id=f"doc{i}_0", credibility=0.85, reliability=0.20)
            for i in range(5)
        ]
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence < 0.80
        assert result.warning_flag is True

    def test_single_doc_multi_chunks_penalty(self):
        evidence = [
            _make_evidence(chunk_id=f"singledoc_{i}", credibility=0.85, reliability=0.75)
            for i in range(5)
        ]
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence < 0.80
        assert result.warning_flag is True

    def test_few_high_cred_sources_penalty(self):
        evidence = [
            _make_evidence(chunk_id=f"doc{i}_0", credibility=0.50, reliability=0.75)
            for i in range(5)
        ]
        result = self.validator.validate(evidence, 0.80)
        assert result.adjusted_confidence < 0.80

    def test_confidence_never_goes_below_zero(self):
        evidence = [
            _make_evidence(chunk_id=f"doc0_{i}", credibility=0.10, reliability=0.10)
            for i in range(1)
        ]
        result = self.validator.validate(evidence, 0.10)
        assert result.adjusted_confidence >= 0.0

    def test_multiple_penalties_stack(self):
        single_low_quality = [
            _make_evidence(chunk_id=f"onlydoc_{i}", credibility=0.20, reliability=0.20)
            for i in range(2)
        ]
        result = self.validator.validate(single_low_quality, 0.90)
        assert result.adjusted_confidence < 0.60
