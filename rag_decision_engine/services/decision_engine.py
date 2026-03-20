from dataclasses import dataclass, field
from typing import Optional
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.contradiction_service import ContradictionReport
logger = get_logger(__name__)
_CONTRADICTION_THRESHOLD = 0.6
_MARGIN_STRONG = 0.10
_MARGIN_WEAK = 0.03
@dataclass
class EngineResult:
    recommendation: Optional[str]
    decision_type: str
    confidence: float
    decision_reason: str
    key_factors: list[str] = field(default_factory=list)
class DecisionEngine:
    def evaluate(
        self,
        option_scores: list[tuple[str, float]],
        contradiction_report: ContradictionReport,
        base_confidence: float,
    ) -> EngineResult:
        if not option_scores:
            return EngineResult(
                recommendation=None,
                decision_type="inconclusive",
                confidence=0.0,
                decision_reason="No options found in the evidence base.",
            )
        sorted_options = sorted(option_scores, key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_options[0]
        second_score = sorted_options[1][1] if len(sorted_options) > 1 else 0.0
        margin = best_score - second_score
        factors: list[str] = []
        if (
            contradiction_report.detected
            and contradiction_report.max_contradiction_score > _CONTRADICTION_THRESHOLD
        ):
            logger.info("engine_inconclusive_contradiction", max_score=contradiction_report.max_contradiction_score)
            return EngineResult(
                recommendation=None,
                decision_type="inconclusive",
                confidence=0.0,
                decision_reason=(
                    f"Evidence is internally contradictory "
                    f"(conflict score {contradiction_report.max_contradiction_score:.2f} exceeds threshold {_CONTRADICTION_THRESHOLD}). "
                    "Manual review required."
                ),
                key_factors=["Strong evidence contradictions detected"],
            )
        if margin < _MARGIN_WEAK and len(sorted_options) > 1:
            logger.info("engine_inconclusive_margin", margin=margin)
            return EngineResult(
                recommendation=None,
                decision_type="inconclusive",
                confidence=0.0,
                decision_reason=(
                    f"Options are too close to differentiate "
                    f"(score gap {margin:.4f} is below minimum {_MARGIN_WEAK}). "
                    "Insufficient evidence differentiation."
                ),
                key_factors=["Score gap too small for confident decision"],
            )
        factors.append(f"{best_name} scores {best_score:.3f} vs {second_score:.3f} runner-up")
        if contradiction_report.detected:
            confidence = max(0.0, base_confidence - 0.10)
            factors.append(f"{contradiction_report.pair_count} minor contradiction(s) detected — confidence reduced")
        else:
            confidence = base_confidence
            factors.append("No contradictions in evidence")
        if margin >= _MARGIN_STRONG:
            decision_type = "strong"
            reason = (
                f"Strong evidence supports {best_name} "
                f"(score {best_score:.3f}, margin {margin:.3f} over next option)."
            )
            factors.append(f"Clear margin of {margin:.3f} — strong signal")
        else:
            decision_type = "weak"
            reason = (
                f"Weak signal favours {best_name} "
                f"(score {best_score:.3f}, narrow margin {margin:.3f}). "
                "Consider gathering more evidence before deciding."
            )
            factors.append(f"Narrow margin {margin:.3f} — weak signal")
        logger.info(
            "engine_result",
            recommendation=best_name,
            decision_type=decision_type,
            confidence=round(confidence, 3),
            margin=round(margin, 4),
        )
        return EngineResult(
            recommendation=best_name,
            decision_type=decision_type,
            confidence=round(confidence, 4),
            decision_reason=reason,
            key_factors=factors,
        )
