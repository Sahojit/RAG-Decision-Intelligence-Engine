import pytest
from rag_decision_engine.services.decision_engine import DecisionEngine, EngineResult
from rag_decision_engine.services.contradiction_service import ContradictionReport, ContradictionPair


def _no_contradiction() -> ContradictionReport:
    return ContradictionReport(detected=False, pairs=[], max_contradiction_score=0.0)


def _with_contradiction(score: float) -> ContradictionReport:
    pair = ContradictionPair(
        chunk_id_a="a", chunk_id_b="b",
        text_a="text a", text_b="text b",
        contradiction_score=score,
    )
    return ContradictionReport(detected=True, pairs=[pair], max_contradiction_score=score)


class TestDecisionEngineEmptyAndSingle:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_empty_options_returns_inconclusive(self):
        result = self.engine.evaluate([], _no_contradiction(), 0.8)
        assert result.recommendation is None
        assert result.decision_type == "inconclusive"
        assert result.confidence == 0.0

    def test_single_option_returns_strong(self):
        result = self.engine.evaluate([("OptionA", 0.9)], _no_contradiction(), 0.8)
        assert result.recommendation == "OptionA"
        assert result.decision_type == "strong"
        assert result.confidence > 0.0


class TestDecisionEngineStrongDecision:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_strong_margin_gives_strong_decision(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.9)
        assert result.recommendation == "OptionA"
        assert result.decision_type == "strong"

    def test_strong_decision_preserves_confidence(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.85)
        assert result.confidence == 0.85

    def test_best_option_wins(self):
        scores = [("B", 0.5), ("A", 0.9), ("C", 0.3)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert result.recommendation == "A"


class TestDecisionEngineWeakDecision:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_narrow_margin_gives_weak_decision(self):
        scores = [("OptionA", 0.50), ("OptionB", 0.45)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert result.recommendation == "OptionA"
        assert result.decision_type == "weak"

    def test_weak_decision_mentions_gather_more_evidence(self):
        scores = [("X", 0.50), ("Y", 0.45)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert "evidence" in result.decision_reason.lower()


class TestDecisionEngineInconclusive:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_contradiction_above_threshold_returns_inconclusive(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        report = _with_contradiction(0.75)
        result = self.engine.evaluate(scores, report, 0.8)
        assert result.recommendation is None
        assert result.decision_type == "inconclusive"
        assert result.confidence == 0.0

    def test_contradiction_at_threshold_returns_inconclusive(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        report = _with_contradiction(0.61)
        result = self.engine.evaluate(scores, report, 0.8)
        assert result.decision_type == "inconclusive"

    def test_margin_below_weak_threshold_returns_inconclusive(self):
        scores = [("OptionA", 0.500), ("OptionB", 0.499)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert result.recommendation is None
        assert result.decision_type == "inconclusive"

    def test_inconclusive_reason_mentions_contradiction(self):
        scores = [("A", 0.8), ("B", 0.5)]
        result = self.engine.evaluate(scores, _with_contradiction(0.8), 0.8)
        assert "contradict" in result.decision_reason.lower()


class TestDecisionEngineContradictionPenalty:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_minor_contradiction_reduces_confidence(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        report = _with_contradiction(0.3)
        result = self.engine.evaluate(scores, report, 0.9)
        assert result.recommendation == "OptionA"
        assert result.confidence < 0.9

    def test_no_contradiction_keeps_confidence(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.9)
        assert result.confidence == 0.9


class TestDecisionEngineKeyFactors:
    def setup_method(self):
        self.engine = DecisionEngine()

    def test_key_factors_populated(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert len(result.key_factors) > 0

    def test_result_is_engine_result_instance(self):
        scores = [("OptionA", 0.80), ("OptionB", 0.60)]
        result = self.engine.evaluate(scores, _no_contradiction(), 0.8)
        assert isinstance(result, EngineResult)
