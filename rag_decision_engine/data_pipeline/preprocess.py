import re
import unicodedata
from typing import Optional
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _remove_control_characters(text)
    text = _collapse_whitespace(text)
    return text.strip()
def extract_metadata_hints(text: str, source_path: str) -> dict[str, object]:
    return {
        "source_type": _infer_source_type(source_path),
        "estimated_year": _extract_year(text),
        "has_citations": _has_citations(text),
        "word_count": len(text.split()),
    }
def _remove_control_characters(text: str) -> str:
    return "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )
def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text
_SOURCE_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"arxiv|ieee|acm|springer|elsevier", re.I), "research_paper"),
    (re.compile(r"docs\.|documentation|readthedocs|devdocs", re.I), "official_documentation"),
    (re.compile(r"medium\.com|towardsdatascience|blog\.", re.I), "technical_blog"),
    (re.compile(r"stackoverflow|reddit|quora|forum", re.I), "forum"),
    (re.compile(r"\.pdf$", re.I), "research_paper"),
]
def _infer_source_type(source_path: str) -> str:
    for pattern, label in _SOURCE_TYPE_PATTERNS:
        if pattern.search(source_path):
            return label
    return "unknown"
_YEAR_RE = re.compile(r"\b(19[89]\d|20[012]\d)\b")
def _extract_year(text: str) -> Optional[int]:
    hits = [int(m.group()) for m in _YEAR_RE.finditer(text[:2000])]
    return max(hits) if hits else None
_CITATION_RE = re.compile(
    r"\[\d+\]"
    r"|\(\w[^)]{2,30}\d{4}\)"
    r"|et al\.",
    re.I,
)
def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text))
