import time
from dataclasses import dataclass
from typing import Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_decision_engine.retrieval.reranker import CrossEncoderReranker
from rag_decision_engine.services.contradiction_service import (
    ContradictionDetector,
    ContradictionReport,
)
from rag_decision_engine.services.decision_policy import DecisionPolicyEngine
from rag_decision_engine.services.reflection_service import ReflectionService
from rag_decision_engine.services.reliability_service import (
    ReliabilityService,
    ScoredEvidence,
)
logger = get_logger(__name__)
class OptionEvidence(BaseModel):
    option: str
    supporting_documents: int = 0
    avg_similarity: float = 0.0
    reliability_score: float = 0.0
    source_credibility: float = 0.0
    final_evidence_score: float = 0.0
    top_evidence_snippets: list[str] = Field(default_factory=list)
class DecisionReport(BaseModel):
    query: str
    options: list[OptionEvidence]
    contradiction_detected: bool
    contradiction_count: int
    contradiction_details: list[dict] = Field(default_factory=list)
    recommended_option: Optional[str]
    recommendation_confidence: float
    policy_reason: str = ""
    reflection_flag: bool = False
    reflection_reason: str = ""
    reasoning: str
    latency_ms: float
    model_used: str = settings.ollama_model
def parse_options(query: str) -> list[str]:
    import re
    pattern = re.compile(r"\bor\b|\bvs\.?\b|\bversus\b", re.I)
    parts = pattern.split(query)
    if len(parts) >= 2:
        cleaned = [p.strip().strip("?.,") for p in parts if p.strip()]
        return cleaned[:4]
    return [query.strip()]
class DecisionService:
    def __init__(self) -> None:
        self._retriever = HybridRetriever()
        self._reranker = CrossEncoderReranker()
        self._reliability = ReliabilityService()
        self._contradiction = ContradictionDetector()
        self._policy = DecisionPolicyEngine()
        self._reflection = ReflectionService()
        self._llm = self._build_llm()
    def decide(self, query: str) -> DecisionReport:
        t_start = time.perf_counter()
        logger.info("decision_start", query=query[:120])
        options = parse_options(query)
        logger.info("options_parsed", options=options)
        option_evidences: list[OptionEvidence] = []
        all_scored: list[ScoredEvidence] = []
        for option in options:
            option_query = f"{query} {option}"
            raw_docs = self._retriever.search(option_query, top_k=settings.rerank_top_k)
            reranked = self._reranker.rerank(option_query, raw_docs)
            scored = self._reliability.score_evidence(reranked)
            all_scored.extend(scored)
            option_evidences.append(self._summarise_option(option, scored))
        contradiction_report = self._contradiction.detect(all_scored)
        option_scores = [
            (oe.option, oe.final_evidence_score) for oe in option_evidences
        ]
        option_doc_counts = {oe.option: oe.supporting_documents for oe in option_evidences}
        _, base_confidence = self._pick_recommendation(option_evidences)
        policy_result = self._policy.evaluate(
            option_scores=option_scores,
            option_doc_counts=option_doc_counts,
            contradiction_report=contradiction_report,
            base_confidence=base_confidence,
        )
        recommendation = policy_result.recommended_option
        confidence = policy_result.adjusted_confidence
        reflection_result = self._reflection.reflect(
            all_evidence=all_scored,
            recommended_option=recommendation,
        )
        confidence = max(
            0.0, round(confidence + reflection_result.confidence_adjustment, 4)
        )
        reasoning = self._generate_reasoning(
            query, option_evidences, contradiction_report, recommendation
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        report = DecisionReport(
            query=query,
            options=option_evidences,
            contradiction_detected=contradiction_report.detected,
            contradiction_count=contradiction_report.pair_count,
            contradiction_details=[
                {
                    "chunk_id_a": p.chunk_id_a,
                    "chunk_id_b": p.chunk_id_b,
                    "snippet_a": p.text_a[:150],
                    "snippet_b": p.text_b[:150],
                    "score": p.contradiction_score,
                }
                for p in contradiction_report.pairs[:5]
            ],
            recommended_option=recommendation,
            recommendation_confidence=confidence,
            policy_reason=policy_result.policy_reason,
            reflection_flag=reflection_result.reflection_flag,
            reflection_reason=reflection_result.reflection_reason,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )
        logger.info(
            "decision_complete",
            recommendation=recommendation,
            confidence=confidence,
            contradiction=contradiction_report.detected,
            policy_inconclusive=policy_result.is_inconclusive,
            reflection_flag=reflection_result.reflection_flag,
            latency_ms=latency_ms,
        )
        return report
    @staticmethod
    def _summarise_option(
        option: str,
        scored: list[ScoredEvidence],
    ) -> OptionEvidence:
        if not scored:
            return OptionEvidence(option=option)
        top_n = scored[:5]
        avg_sim = sum(e.similarity_score for e in top_n) / len(top_n)
        avg_rel = sum(e.reliability_score for e in top_n) / len(top_n)
        avg_cred = sum(e.source_credibility for e in top_n) / len(top_n)
        avg_final = sum(e.final_score for e in top_n) / len(top_n)
        return OptionEvidence(
            option=option,
            supporting_documents=len(scored),
            avg_similarity=round(avg_sim, 4),
            reliability_score=round(avg_rel, 4),
            source_credibility=round(avg_cred, 4),
            final_evidence_score=round(avg_final, 4),
            top_evidence_snippets=[e.text[:200] for e in top_n],
        )
    @staticmethod
    def _pick_recommendation(
        options: list[OptionEvidence],
    ) -> tuple[Optional[str], float]:
        if not options:
            return None, 0.0
        best = max(options, key=lambda o: o.final_evidence_score)
        scores = [o.final_evidence_score for o in options]
        total = sum(scores)
        confidence = best.final_evidence_score / total if total > 0 else 0.0
        return best.option, confidence
    def _generate_reasoning(
        self,
        query: str,
        options: list[OptionEvidence],
        contradiction: ContradictionReport,
        recommendation: Optional[str],
    ) -> str:
        if self._llm is None:
            return self._fallback_reasoning(options, contradiction, recommendation)
        options_text = "\n".join(
            f"- {o.option}: evidence_score={o.final_evidence_score}, "
            f"reliability={o.reliability_score}, credibility={o.source_credibility}"
            for o in options
        )
        contradiction_text = (
            f"YES — {contradiction.pair_count} contradicting pairs detected"
            if contradiction.detected
            else "NO contradictions detected"
        )
        prompt = PromptTemplate(
            input_variables=["query", "options", "contradiction", "recommendation"],
            template="""You are an expert decision analyst. Based on the evidence analysis below,
provide a concise 3-5 sentence reasoning for the recommendation.

Query: {query}

Evidence Summary:
{options}

Contradictions: {contradiction}

Recommended Decision: {recommendation}

Reasoning (be specific and evidence-based):""",
        )
        chain = LLMChain(llm=self._llm, prompt=prompt)
        try:
            result = chain.run(
                query=query,
                options=options_text,
                contradiction=contradiction_text,
                recommendation=recommendation or "Insufficient evidence",
            )
            return result.strip()
        except Exception as exc:
            logger.warning("llm_reasoning_failed", error=str(exc))
            return self._fallback_reasoning(options, contradiction, recommendation)
    @staticmethod
    def _fallback_reasoning(
        options: list[OptionEvidence],
        contradiction: ContradictionReport,
        recommendation: Optional[str],
    ) -> str:
        if not options:
            return "Insufficient evidence to make a recommendation."
        best = max(options, key=lambda o: o.final_evidence_score)
        lines = [
            f"Based on {best.supporting_documents} supporting documents, "
            f"'{best.option}' achieves the highest evidence score "
            f"({best.final_evidence_score:.2f}) combining similarity, "
            f"reliability ({best.reliability_score:.2f}), and "
            f"source credibility ({best.source_credibility:.2f})."
        ]
        if contradiction.detected:
            lines.append(
                f"Note: {contradiction.pair_count} contradicting evidence pair(s) were detected. "
                "Review the contradiction details before acting on this recommendation."
            )
        return " ".join(lines)
    @staticmethod
    def _build_llm() -> Optional[object]:
        try:
            llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.llm_temperature,
            )
            return llm
        except Exception as exc:
            logger.warning("ollama_unavailable", error=str(exc))
            return None
