import re
from dataclasses import dataclass
from typing import Optional
from rag_decision_engine.config.logging_config import get_logger
logger = get_logger(__name__)
_CURRENT_YEAR = 2026
@dataclass
class QueryFilters:
    source_type: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    has_citations: Optional[bool] = None
    def is_empty(self) -> bool:
        return (
            self.source_type is None
            and self.min_year is None
            and self.max_year is None
            and self.has_citations is None
        )
    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "min_year": self.min_year,
            "max_year": self.max_year,
            "has_citations": self.has_citations,
        }
_SOURCE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bresearch\s+papers?\b", re.I), "research_paper"),
    (re.compile(r"\bacademic\s+papers?\b", re.I), "research_paper"),
    (re.compile(r"\bjournal\s+articles?\b", re.I), "research_paper"),
    (re.compile(r"\bpapers?\b", re.I), "research_paper"),
    (re.compile(r"\bblog\s+posts?\b|\bblogs?\b", re.I), "blog"),
    (re.compile(r"\bdocumentation\b|\bdocs?\b", re.I), "documentation"),
    (re.compile(r"\bforum[s]?\b|\bdiscussions?\b|\bstackoverflow\b", re.I), "forum"),
]
_YEAR_AFTER = re.compile(r"\b(?:after|since|from)\s+(20\d{2})\b", re.I)
_YEAR_BEFORE = re.compile(r"\b(?:before|until|up\s+to)\s+(20\d{2})\b", re.I)
_YEAR_EXACT = re.compile(r"\bin\s+(20\d{2})\b", re.I)
_YEAR_RANGE = re.compile(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b")
_RECENT = re.compile(r"\b(?:recent|latest|newest|new|modern|current)\b", re.I)
_CITATIONS = re.compile(r"\bwith\s+citations?\b|\bhighly\s+cited\b|\bcited\b|\bpeer[\s-]?reviewed\b", re.I)
def extract_filters(query: str) -> QueryFilters:
    filters = QueryFilters()
    for pattern, source_type in _SOURCE_MAP:
        if pattern.search(query):
            filters.source_type = source_type
            break
    m = _YEAR_RANGE.search(query)
    if m:
        filters.min_year = int(m.group(1))
        filters.max_year = int(m.group(2))
    else:
        m = _YEAR_AFTER.search(query)
        if m:
            filters.min_year = int(m.group(1))
        m = _YEAR_BEFORE.search(query)
        if m:
            filters.max_year = int(m.group(1))
        m = _YEAR_EXACT.search(query)
        if m:
            y = int(m.group(1))
            filters.min_year = y
            filters.max_year = y
    if _RECENT.search(query) and filters.min_year is None:
        filters.min_year = _CURRENT_YEAR - 3
    if _CITATIONS.search(query):
        filters.has_citations = True
    if not filters.is_empty():
        logger.info(
            "query_filters_extracted",
            query_preview=query[:80],
            filters=filters.to_dict(),
        )
    return filters
