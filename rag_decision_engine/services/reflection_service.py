from dataclasses import dataclass, field
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.reliability_service import ScoredEvidence
logger = get_logger(__name__)
_MIN_EVIDENCE_TOTAL = 3
_MIN_AVG_CREDIBILITY = 0.60
_HIGH_CREDIBILITY_MIN = 0.80
_HIGH_CRED_RATIO_MIN = 0.40
_MAX_CREDIBILITY_SPREAD = 0.35
_MIN_RELIABILITY_AVG = 0.50
@dataclass
class ReflectionResult:
    reflection_flag: bool
    reflection_reason: str
    confidence_adjustment: float
    issues: list[str] = field(default_factory=list)
class ReflectionService:
    def reflect(
        self,
        all_evidence: list[ScoredEvidence],
        recommended_option: str | None,
    ) -> ReflectionResult:
        if not all_evidence:
            return ReflectionResult(
                reflection_flag=True,
                reflection_reason="Reflection: no evidence was retrieved for this query.",
                confidence_adjustment=-0.20,
                issues=["no_evidence"],
            )
        issues: list[str] = []
        adjustments: list[float] = []
        credibilities = [e.source_credibility for e in all_evidence]
        reliabilities = [e.reliability_score for e in all_evidence]
        avg_credibility = sum(credibilities) / len(credibilities)
        avg_reliability = sum(reliabilities) / len(reliabilities)
        max_cred = max(credibilities)
        min_cred = min(credibilities)
        cred_spread = max_cred - min_cred
        if len(all_evidence) < _MIN_EVIDENCE_TOTAL:
            issues.append(
                f"sparse_evidence: only {len(all_evidence)} chunk(s) retrieved "
                f"(minimum recommended: {_MIN_EVIDENCE_TOTAL})"
            )
            adjustments.append(-0.10)
        if avg_credibility < _MIN_AVG_CREDIBILITY:
            issues.append(
                f"low_avg_credibility: mean source credibility is "
                f"{avg_credibility:.2f} (threshold: {_MIN_AVG_CREDIBILITY:.2f})"
            )
            adjustments.append(-0.08)
        high_cred_count = sum(
            1 for c in credibilities if c >= _HIGH_CREDIBILITY_MIN
        )
        high_cred_ratio = high_cred_count / len(credibilities)
        if high_cred_ratio < _HIGH_CRED_RATIO_MIN:
            issues.append(
                f"low_high_credibility_ratio: only {high_cred_ratio:.0%} of evidence "
                f"comes from high-credibility sources (≥{_HIGH_CREDIBILITY_MIN}); "
                f"majority may be blog posts or forum content"
            )
            adjustments.append(-0.08)
        if cred_spread > _MAX_CREDIBILITY_SPREAD and len(all_evidence) > 2:
            issues.append(
                f"credibility_spread: credibility ranges from {min_cred:.2f} to "
                f"{max_cred:.2f} (spread: {cred_spread:.2f}), suggesting mixed-quality sources"
            )
            adjustments.append(-0.05)
        if avg_reliability < _MIN_RELIABILITY_AVG:
            issues.append(
                f"low_avg_reliability: mean reliability score is "
                f"{avg_reliability:.2f} (threshold: {_MIN_RELIABILITY_AVG:.2f}); "
                "evidence may not be well-supported by the ML reliability model"
            )
            adjustments.append(-0.07)
        unique_sources = set(
            e.chunk_id.rsplit("_", 1)[0] for e in all_evidence
        )
        if len(unique_sources) == 1 and len(all_evidence) > 2:
            issues.append(
                "single_source_concentration: all retrieved evidence originates "
                "from a single document — recommendation lacks diverse source validation"
            )
            adjustments.append(-0.10)
        total_adjustment = sum(adjustments)
        flag = bool(issues)
        if flag:
            issue_summary = "; ".join(issues)
            reason = f"Reflection warning: {issue_summary}."
        else:
            reason = (
                f"Reflection passed — {len(all_evidence)} evidence chunks, "
                f"avg credibility {avg_credibility:.2f}, "
                f"avg reliability {avg_reliability:.2f}. "
                "No quality concerns detected."
            )
        logger.info(
            "reflection_complete",
            flag=flag,
            issues=len(issues),
            confidence_adjustment=round(total_adjustment, 3),
            avg_credibility=round(avg_credibility, 3),
            avg_reliability=round(avg_reliability, 3),
        )
        return ReflectionResult(
            reflection_flag=flag,
            reflection_reason=reason,
            confidence_adjustment=round(total_adjustment, 4),
            issues=issues,
        )
