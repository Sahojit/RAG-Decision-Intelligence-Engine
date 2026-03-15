import pytest
from rag_decision_engine.data_pipeline.preprocess import (
    clean_text,
    extract_metadata_hints,
    _infer_source_type,
    _extract_year,
    _has_citations,
)
class TestCleanText:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"
    def test_collapses_multiple_spaces(self):
        result = clean_text("hello    world")
        assert "  " not in result
    def test_removes_null_bytes(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result
    def test_preserves_newlines(self):
        result = clean_text("line one\nline two")
        assert "\n" in result
    def test_collapses_excessive_blank_lines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result
    def test_unicode_normalisation(self):
        result = clean_text("\ufb01le")
        assert result == "file"
class TestInferSourceType:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("https://arxiv.org/abs/123", "research_paper"),
            ("https://docs.python.org/3/", "official_documentation"),
            ("https://medium.com/post", "technical_blog"),
            ("https://stackoverflow.com/q/1", "forum"),
            ("/data/paper.pdf", "research_paper"),
            ("/data/unknown.xyz", "unknown"),
        ],
    )
    def test_source_type(self, path: str, expected: str):
        assert _infer_source_type(path) == expected
class TestExtractYear:
    def test_extracts_recent_year(self):
        assert _extract_year("Published in 2023.") == 2023
    def test_returns_most_recent(self):
        assert _extract_year("From 2019 to 2024.") == 2024
    def test_returns_none_when_no_year(self):
        assert _extract_year("No date here.") is None
class TestHasCitations:
    def test_detects_bracket_citation(self):
        assert _has_citations("See [1] for details.") is True
    def test_detects_et_al(self):
        assert _has_citations("Smith et al. showed.") is True
    def test_no_citations(self):
        assert _has_citations("This has no references.") is False
class TestExtractMetadataHints:
    def test_returns_required_keys(self):
        hints = extract_metadata_hints("Text from 2022 [1].", "paper.pdf")
        assert "source_type" in hints
        assert "estimated_year" in hints
        assert "has_citations" in hints
        assert "word_count" in hints
