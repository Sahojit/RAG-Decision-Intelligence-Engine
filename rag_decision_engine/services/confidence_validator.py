from dataclasses import dataclass
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.reliability_service import ScoredEvidence
logger = get_logger(__name__)
_MIN_EVIDENCE = 3
_MIN_AVG_CREDIBILITY = 0.60
_HIGH_CRED_THRESHOLD = 0.80
_HIGH_CRED_MIN_RATIO = 0.30
_MIN_AVG_RELIABILITY = 0.45
@dataclass
class ValidationResult:
    adjusted_confidence: float
    validation_reason: str
    warning_flag: bool
class ConfidenceValidator:
    def validate(
        self,
        all_evidence: list[ScoredEvidence],
        base_confidence: float,
    ) -> ValidationResult:
        if not all_evidence:
            return ValidationResult(
                adjusted_confidence=0.0,
                validation_reason="No evidence retrieved — confidence set to zero.",
                warning_flag=True,
            )
        warnings: list[str] = []
        penalty = 0.0
        credibilities = [e.source_credibility for e in all_evidence]
        reliabilities = [e.reliability_score for e in all_evidence]
        avg_cred = sum(credibilities) / len(credibilities)
        avg_rel = sum(reliabilities) / len(reliabilities)
        high_cred_ratio = sum(1 for c in credibilities if c >= _HIGH_CRED_THRESHOLD) / len(credibilities)
        if len(all_evidence) < _MIN_EVIDENCE:
            warnings.append(f"sparse evidence ({len(all_evidence)} chunks, min {_MIN_EVIDENCE})")
            penalty += 0.12
        if avg_cred < _MIN_AVG_CREDIBILITY:
            warnings.append(f"low source credibility (avg {avg_cred:.2f})")
            penalty += 0.08
        if high_cred_ratio < _HIGH_CRED_MIN_RATIO:
            warnings.append(f"few high-quality sources ({high_cred_ratio:.0%} from research/docs)")
            penalty += 0.07
        if avg_rel < _MIN_AVG_RELIABILITY:
            warnings.append(f"low reliability score (avg {avg_rel:.2f})")
            penalty += 0.06
        unique_docs = len(set(e.chunk_id.rsplit("_", 1)[0] for e in all_evidence))
        if unique_docs == 1 and len(all_evidence) > 2:
            warnings.append("all evidence from a single document")
            penalty += 0.10
        adjusted = round(max(0.0, base_confidence - penalty), 4)
        flag = bool(warnings)
        if flag:
            reason = "Warning: " + "; ".join(warnings) + "."
        else:
            reason = (
                f"Evidence quality OK — {len(all_evidence)} chunks, "
                f"avg credibility {avg_cred:.2f}, avg reliability {avg_rel:.2f}."
            )
        logger.info(
            "confidence_validation",
            base=round(base_confidence, 3),
            adjusted=adjusted,
            penalty=round(penalty, 3),
            warnings=len(warnings),
        )
        return ValidationResult(
            adjusted_confidence=adjusted,
            validation_reason=reason,
            warning_flag=flag,
        )
