import re
from rag_decision_engine.config.logging_config import get_logger
logger = get_logger(__name__)
_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpolar\b", re.I), "polars"),
    (re.compile(r"\bpyspark\b", re.I), "pyspark"),
    (re.compile(r"\bscikit learn\b", re.I), "scikit-learn"),
    (re.compile(r"\btf\b", re.I), "tensorflow"),
    (re.compile(r"\bpt\b", re.I), "pytorch"),
]
def normalize_query(query: str) -> str:
    normalized = query
    for pattern, replacement in _REPLACEMENTS:
        result = pattern.sub(replacement, normalized)
        if result != normalized:
            logger.debug("query_term_normalized", before=normalized[:80], replacement=replacement)
            normalized = result
    return normalized
