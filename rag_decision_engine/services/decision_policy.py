from dataclasses import dataclass
from typing import Optional
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.contradiction_service import ContradictionReport
logger = get_logger(__name__)
_CONTRADICTION_HARD_BLOCK = 0.6
_MARGIN_MIN = 0.05
_MIN_SUPPORTING_DOCS = 3
@dataclass
class PolicyResult:
    recommended_option: Optional[str]
    adjusted_confidence: float
    policy_reason: str
    is_inconclusive: bool
class DecisionPolicyEngine:
    def evaluate(
        self,
        option_scores: list[tuple[str, float]],
        option_doc_counts: dict[str, int],
        contradiction_report: ContradictionReport,
        base_confidence: float,
    ) -> PolicyResult:
        if not option_scores:
            return PolicyResult(
                recommended_option=None,
                adjusted_confidence=0.0,
                policy_reason="No options found in the evidence base.",
                is_inconclusive=True,
            )
        sorted_options = sorted(option_scores, key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_options[0]
        second_score = sorted_options[1][1] if len(sorted_options) > 1 else 0.0
        if (
            contradiction_report.detected
            and contradiction_report.max_contradiction_score > _CONTRADICTION_HARD_BLOCK
        ):
            reason = (
                f"Inconclusive — strong contradictions detected "
                f"(max contradiction score: {contradiction_report.max_contradiction_score:.2f}, "
                f"threshold: {_CONTRADICTION_HARD_BLOCK}). "
                "Evidence is internally inconsistent; manual review required."
            )
            logger.info(
                "policy_hard_contradiction_block",
                max_score=contradiction_report.max_contradiction_score,
            )
            return PolicyResult(
                recommended_option=None,
                adjusted_confidence=0.0,
                policy_reason=reason,
                is_inconclusive=True,
            )
        margin = best_score - second_score
        if margin < _MARGIN_MIN and len(sorted_options) > 1:
            reason = (
                f"Inconclusive — score margin between top options is too small "
                f"('{best_name}' score: {best_score:.4f}, "
                f"runner-up score: {second_score:.4f}, "
                f"margin: {margin:.4f} < required {_MARGIN_MIN}). "
                "Insufficient evidence differentiation to make a confident recommendation."
            )
            logger.info(
                "policy_insufficient_margin",
                best=best_name,
                best_score=best_score,
                second_score=second_score,
                margin=margin,
            )
            return PolicyResult(
                recommended_option=None,
                adjusted_confidence=0.0,
                policy_reason=reason,
                is_inconclusive=True,
            )
        best_doc_count = option_doc_counts.get(best_name, 0)
        adjusted_confidence = base_confidence
        confidence_notes: list[str] = []
        if best_doc_count < _MIN_SUPPORTING_DOCS:
            penalty = 0.15
            adjusted_confidence = max(0.0, base_confidence - penalty)
            confidence_notes.append(
                f"confidence penalised by {penalty:.0%} "
                f"(only {best_doc_count} supporting document(s), "
                f"minimum recommended: {_MIN_SUPPORTING_DOCS})"
            )
        if contradiction_report.detected:
            soft_penalty = 0.10
            adjusted_confidence = max(0.0, adjusted_confidence - soft_penalty)
            confidence_notes.append(
                f"confidence further penalised by {soft_penalty:.0%} "
                f"due to {contradiction_report.pair_count} contradicting evidence pair(s)"
            )
        if confidence_notes:
            reason = (
                f"Recommending '{best_name}' with reduced confidence. "
                + "; ".join(confidence_notes).capitalize() + "."
            )
        else:
            reason = (
                f"Recommending '{best_name}' — highest evidence score "
                f"({best_score:.4f}) with a clear margin of {margin:.4f} over the next option."
            )
        logger.info(
            "policy_recommendation",
            option=best_name,
            base_confidence=round(base_confidence, 3),
            adjusted_confidence=round(adjusted_confidence, 3),
            margin=round(margin, 4),
        )
        return PolicyResult(
            recommended_option=best_name,
            adjusted_confidence=round(adjusted_confidence, 4),
            policy_reason=reason,
            is_inconclusive=False,
        )
