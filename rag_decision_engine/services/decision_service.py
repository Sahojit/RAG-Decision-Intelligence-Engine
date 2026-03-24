import time
from typing import Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_decision_engine.retrieval.reranker import CrossEncoderReranker
from rag_decision_engine.services.confidence_validator import ConfidenceValidator
from rag_decision_engine.services.contradiction_service import ContradictionDetector
from rag_decision_engine.services.decision_engine import DecisionEngine
from rag_decision_engine.services.evidence_service import (
    OptionEvidence,
    format_evidence_for_prompt,
    select_top_evidence_batch,
)
from rag_decision_engine.services.live_retrieval import detect_query_type, fetch_live_documents_sync
from rag_decision_engine.services.query_filter_parser import QueryFilters, extract_filters
from rag_decision_engine.services.query_normalizer import normalize_query
from rag_decision_engine.services.reliability_service import ReliabilityService, ScoredEvidence
logger = get_logger(__name__)
_QUERY_CACHE: dict[str, "DecisionReport"] = {}
_CACHE_MAX = 100
class OptionResult(BaseModel):
    name: str
    evidence_score: float
    reliability: float
    credibility: float
    supporting_docs: int
    top_snippets: list[str] = Field(default_factory=list)
class SourcesSummary(BaseModel):
    research_papers: int = 0
    documentation: int = 0
    blogs: int = 0
    forums: int = 0
    live_api: int = 0
class DecisionReport(BaseModel):
    query: str
    options: list[OptionResult]
    recommendation: Optional[str]
    decision_type: str
    confidence: float
    key_factors: list[str] = Field(default_factory=list)
    contradictions: int
    sources_summary: SourcesSummary
    evidence: list[OptionEvidence] = Field(default_factory=list)
    reasoning: str
    latency_ms: float
    live_retrieval_used: bool = False
    live_docs_count: int = 0
    local_docs_count: int = 0
    filters_applied: dict = Field(default_factory=dict)
    validation_warning: bool = False
    validation_reason: str = ""
def _parse_options(query: str) -> list[str]:
    import re
    pattern = re.compile(r"\bor\b|\bvs\.?\b|\bversus\b", re.I)
    parts = pattern.split(query)
    if len(parts) >= 2:
        return [p.strip().strip("?.,") for p in parts if p.strip()][:4]
    return [query.strip()]
def _build_sources_summary(all_scored: list[ScoredEvidence], live_docs_count: int) -> SourcesSummary:
    summary = SourcesSummary()
    live_seen = 0
    for e in all_scored:
        origin = e.metadata.get("origin", "")
        if origin in ("arxiv", "semantic_scholar") and live_seen < live_docs_count:
            summary.live_api += 1
            live_seen += 1
            continue
        src = str(e.metadata.get("source_type", "unknown"))
        if src == "research_paper":
            summary.research_papers += 1
        elif src == "documentation":
            summary.documentation += 1
        elif src == "blog":
            summary.blogs += 1
        elif src == "forum":
            summary.forums += 1
    return summary
def _summarise_option(option: str, scored: list[ScoredEvidence]) -> OptionResult:
    if not scored:
        return OptionResult(name=option, evidence_score=0.0, reliability=0.0, credibility=0.0, supporting_docs=0)
    top = scored[:5]
    return OptionResult(
        name=option,
        evidence_score=round(sum(e.final_score for e in top) / len(top), 4),
        reliability=round(sum(e.reliability_score for e in top) / len(top), 4),
        credibility=round(sum(e.source_credibility for e in top) / len(top), 4),
        supporting_docs=len(scored),
        top_snippets=[e.text[:200] for e in top],
    )
class DecisionService:
    def __init__(self) -> None:
        self._retriever = HybridRetriever()
        self._reranker = CrossEncoderReranker()
        self._reliability = ReliabilityService()
        self._contradiction = ContradictionDetector()
        self._engine = DecisionEngine()
        self._validator = ConfidenceValidator()
        self._llm = self._build_llm()
    def decide(self, query: str, use_live_retrieval: bool = False) -> DecisionReport:
        query = normalize_query(query)
        is_research = detect_query_type(query)
        effective_live = use_live_retrieval or is_research
        cache_key = f"{query}|{effective_live}"
        if cache_key in _QUERY_CACHE:
            logger.info("cache_hit", query=query[:60])
            return _QUERY_CACHE[cache_key]
        t_start = time.perf_counter()
        logger.info("decision_start", query=query[:120], live_retrieval=effective_live, is_research=is_research)
        filters = extract_filters(query)
        if is_research and filters.source_type is None:
            filters = QueryFilters(
                source_type="research_paper",
                min_year=filters.min_year or 2018,
                max_year=filters.max_year,
                has_citations=filters.has_citations,
            )
        options = _parse_options(query)
        live_docs = []
        if effective_live:
            live_docs = fetch_live_documents_sync(query)
            logger.info("live_docs_fetched", count=len(live_docs))
        option_results: list[OptionResult] = []
        all_scored: list[ScoredEvidence] = []
        option_docs_map: dict[str, list[ScoredEvidence]] = {}
        for option in options:
            option_query = f"{query} {option}"
            raw_docs = self._retriever.search(
                option_query,
                filters=filters,
                dynamic_docs=live_docs if live_docs else None,
            )
            reranked = self._reranker.rerank(option_query, raw_docs)
            scored = self._reliability.score_evidence(reranked)
            all_scored.extend(scored)
            option_docs_map[option] = scored
            option_results.append(_summarise_option(option, scored))
        cross_encoder_model = self._reranker._get_model() if self._reranker else None
        option_evidence = select_top_evidence_batch(
            option_docs_map=option_docs_map,
            query=query,
            top_k=3,
            model=cross_encoder_model,
        )
        contradiction_report = self._contradiction.detect(all_scored)
        total_score = sum(r.evidence_score for r in option_results)
        best_score = max((r.evidence_score for r in option_results), default=0.0)
        base_confidence = best_score / total_score if total_score > 0 else 0.0
        engine_result = self._engine.evaluate(
            option_scores=[(r.name, r.evidence_score) for r in option_results],
            contradiction_report=contradiction_report,
            base_confidence=base_confidence,
        )
        validation_result = self._validator.validate(
            all_evidence=all_scored,
            base_confidence=engine_result.confidence,
        )
        reasoning = self._generate_reasoning(
            query=query,
            options=option_results,
            option_evidence=option_evidence,
            recommendation=engine_result.recommendation,
            decision_type=engine_result.decision_type,
            contradiction_count=contradiction_report.pair_count,
        )
        report = DecisionReport(
            query=query,
            options=option_results,
            recommendation=engine_result.recommendation,
            decision_type=engine_result.decision_type,
            confidence=validation_result.adjusted_confidence,
            key_factors=engine_result.key_factors,
            contradictions=contradiction_report.pair_count,
            sources_summary=_build_sources_summary(all_scored, len(live_docs)),
            evidence=option_evidence,
            reasoning=reasoning,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
            live_retrieval_used=effective_live and bool(live_docs),
            live_docs_count=len(live_docs),
            local_docs_count=max(0, len(all_scored) - len(live_docs)),
            filters_applied=filters.to_dict() if not filters.is_empty() else {},
            validation_warning=validation_result.warning_flag,
            validation_reason=validation_result.validation_reason,
        )
        logger.info(
            "decision_complete",
            recommendation=engine_result.recommendation,
            decision_type=engine_result.decision_type,
            confidence=validation_result.adjusted_confidence,
            latency_ms=report.latency_ms,
        )
        if len(_QUERY_CACHE) >= _CACHE_MAX:
            del _QUERY_CACHE[next(iter(_QUERY_CACHE))]
        _QUERY_CACHE[cache_key] = report
        return report
    def _generate_reasoning(
        self,
        query: str,
        options: list[OptionResult],
        option_evidence: list[OptionEvidence],
        recommendation: Optional[str],
        decision_type: str,
        contradiction_count: int,
    ) -> str:
        if self._llm is None:
            return self._fallback_reasoning(options, option_evidence, recommendation, decision_type, contradiction_count)
        evidence_block = format_evidence_for_prompt(option_evidence)
        prompt = PromptTemplate(
            input_variables=["query", "evidence_block", "recommendation", "decision_type", "contradictions"],
            template="""You are an expert tech decision analyst.

STRICT INSTRUCTIONS:
- Base your reasoning ONLY on the evidence quotes provided below.
- Reference the quotes explicitly in your response.
- Do NOT introduce external knowledge or assumptions.
- Keep reasoning to 3-5 sentences.

Query: {query}

Evidence:
{evidence_block}

Decision: {recommendation} ({decision_type})
Contradictions detected: {contradictions}

Reasoning (cite the evidence explicitly):""",
        )
        chain = prompt | self._llm
        try:
            result = chain.invoke({
                "query": query,
                "evidence_block": evidence_block,
                "recommendation": recommendation or "Inconclusive",
                "decision_type": decision_type,
                "contradictions": contradiction_count,
            })
            return result.strip() if isinstance(result, str) else str(result).strip()
        except Exception as exc:
            logger.warning("llm_reasoning_failed", error=str(exc))
            return self._fallback_reasoning(options, option_evidence, recommendation, decision_type, contradiction_count)
    @staticmethod
    def _fallback_reasoning(
        options: list[OptionResult],
        option_evidence: list[OptionEvidence],
        recommendation: Optional[str],
        decision_type: str,
        contradiction_count: int,
    ) -> str:
        if decision_type == "inconclusive" or not recommendation:
            return (
                "The evidence base does not provide a clear differentiation between options. "
                + (f"{contradiction_count} contradicting signals detected. " if contradiction_count else "")
                + "Consider gathering more targeted evidence before deciding."
            )
        best_opt = next((o for o in options if o.name == recommendation), None)
        best_ev = next((oe for oe in option_evidence if oe.option == recommendation), None)
        if not best_opt:
            return "Recommendation based on aggregate evidence scores."
        lines = [
            f"Based on {best_opt.supporting_docs} supporting documents, '{best_opt.name}' achieves "
            f"the highest evidence score ({best_opt.evidence_score:.2f}) with reliability {best_opt.reliability:.2f} "
            f"and source credibility {best_opt.credibility:.2f}."
        ]
        if best_ev and best_ev.items:
            top = best_ev.items[0]
            lines.append(
                f'Key evidence: "{top.quote}" '
                f"({top.source_type}, {top.year or 'n/a'}, score {top.score:.2f})."
            )
        if contradiction_count:
            lines.append(f"{contradiction_count} contradicting evidence pair(s) detected — confidence adjusted.")
        if decision_type == "weak":
            lines.append("Margin over alternatives is narrow — validate with additional domain sources.")
        return " ".join(lines)
    @staticmethod
    def _build_llm() -> Optional[object]:
        try:
            return Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.llm_temperature,
            )
        except Exception as exc:
            logger.warning("ollama_unavailable", error=str(exc))
            return None
