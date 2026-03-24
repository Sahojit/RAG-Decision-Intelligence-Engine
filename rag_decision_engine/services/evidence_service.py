import re
import time
from typing import Optional
import numpy as np
from pydantic import BaseModel
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.reliability_service import ScoredEvidence
logger = get_logger(__name__)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_SENTENCES = 10
_MAX_TEXT_LEN = 3000
_QUOTE_MAX = 280
_TOP_K_DEFAULT = 3
class EvidenceItem(BaseModel):
    quote: str
    source_type: str
    year: Optional[int]
    citations: int = 0
    score: float
    source_origin: str
    chunk_id: str
class OptionEvidence(BaseModel):
    option: str
    items: list[EvidenceItem]
def _split_sentences(text: str) -> list[str]:
    truncated = text[:_MAX_TEXT_LEN]
    raw = _SENT_SPLIT.split(truncated)
    return [s.strip() for s in raw if len(s.strip()) > 20][:_MAX_SENTENCES]
def _truncate_quote(sentence: str) -> str:
    if len(sentence) <= _QUOTE_MAX:
        return sentence
    cut = sentence[:_QUOTE_MAX]
    last_space = cut.rfind(" ")
    return (cut[:last_space] if last_space > 0 else cut) + "…"
def extract_best_sentence(text: str, query: str, model: Optional[object] = None) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _truncate_quote(text[:_QUOTE_MAX])
    if len(sentences) == 1:
        return _truncate_quote(sentences[0])
    if model is not None:
        try:
            pairs = [(query, s) for s in sentences]
            raw = model.predict(pairs, show_progress_bar=False)
            scores = [float(1.0 / (1.0 + np.exp(-r))) for r in raw]
            best = sentences[int(np.argmax(scores))]
            return _truncate_quote(best)
        except Exception as exc:
            logger.warning("sentence_scoring_failed", error=str(exc))
    return _truncate_quote(sentences[0])
def select_top_evidence(
    docs: list[ScoredEvidence],
    query: str,
    top_k: int = _TOP_K_DEFAULT,
    model: Optional[object] = None,
) -> list[EvidenceItem]:
    if not docs:
        return []
    top = sorted(docs, key=lambda d: d.final_score, reverse=True)[:top_k]
    items: list[EvidenceItem] = []
    for doc in top:
        quote = extract_best_sentence(doc.text, query, model)
        if not quote:
            continue
        meta = doc.metadata
        raw_year = meta.get("estimated_year") or meta.get("year")
        try:
            year = int(raw_year) if raw_year is not None else None
        except (ValueError, TypeError):
            year = None
        citations = int(meta.get("citation_count", 0) or 0)
        origin = str(meta.get("origin", ""))
        source_origin = "api" if origin in ("arxiv", "semantic_scholar") else "local"
        items.append(
            EvidenceItem(
                quote=quote,
                source_type=str(meta.get("source_type", "unknown")),
                year=year,
                citations=citations,
                score=round(doc.final_score, 4),
                source_origin=source_origin,
                chunk_id=doc.chunk_id,
            )
        )
    return items
def select_top_evidence_batch(
    option_docs_map: dict[str, list[ScoredEvidence]],
    query: str,
    top_k: int = _TOP_K_DEFAULT,
    model: Optional[object] = None,
) -> list[OptionEvidence]:
    t0 = time.perf_counter()
    results: list[OptionEvidence] = []
    for option, docs in option_docs_map.items():
        items = select_top_evidence(docs, query, top_k=top_k, model=model)
        results.append(OptionEvidence(option=option, items=items))
    logger.info(
        "evidence_attribution_complete",
        options=len(results),
        total_quotes=sum(len(r.items) for r in results),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return results
def format_evidence_for_prompt(option_evidence: list[OptionEvidence]) -> str:
    lines: list[str] = []
    for oe in option_evidence:
        lines.append(f"[Option: {oe.option}]")
        for i, item in enumerate(oe.items, 1):
            cit = f", citations: {item.citations}" if item.citations else ""
            meta = f"{item.source_type}, {item.year or 'n/a'}{cit}, score {item.score:.2f}"
            lines.append(f'{i}. "{item.quote}" ({meta})')
        lines.append("")
    return "\n".join(lines).strip()
