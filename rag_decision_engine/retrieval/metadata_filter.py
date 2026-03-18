from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
from rag_decision_engine.services.query_filter_parser import QueryFilters
logger = get_logger(__name__)
_FALLBACK_MIN = 5
def apply_metadata_filter(
    documents: list[RetrievedDocument],
    filters: QueryFilters,
) -> list[RetrievedDocument]:
    if not documents or filters.is_empty():
        return documents
    original_count = len(documents)
    filtered = [d for d in documents if _passes(d, filters)]
    if len(filtered) < _FALLBACK_MIN:
        logger.warning(
            "metadata_filter_fallback",
            reason="too_few_results",
            filtered=len(filtered),
            total=original_count,
            fallback_min=_FALLBACK_MIN,
        )
        return documents
    logger.info(
        "metadata_filter_applied",
        before=original_count,
        after=len(filtered),
        dropped=original_count - len(filtered),
    )
    return filtered
def _passes(doc: RetrievedDocument, filters: QueryFilters) -> bool:
    meta = doc.metadata
    if filters.source_type is not None:
        if meta.get("source_type") != filters.source_type:
            return False
    if filters.min_year is not None or filters.max_year is not None:
        raw_year = meta.get("estimated_year") or meta.get("year")
        if raw_year is not None:
            try:
                year = int(raw_year)
            except (ValueError, TypeError):
                year = None
            if year is not None:
                if filters.min_year is not None and year < filters.min_year:
                    return False
                if filters.max_year is not None and year > filters.max_year:
                    return False
    if filters.has_citations is True:
        if not meta.get("has_citations"):
            return False
    return True
