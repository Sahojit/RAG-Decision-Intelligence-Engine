import pytest
from rag_decision_engine.services.query_filter_parser import extract_filters, QueryFilters
from rag_decision_engine.services.query_normalizer import normalize_query


class TestQueryFiltersDataclass:
    def test_is_empty_when_all_none(self):
        f = QueryFilters()
        assert f.is_empty() is True

    def test_not_empty_when_source_set(self):
        f = QueryFilters(source_type="arxiv")
        assert f.is_empty() is False

    def test_not_empty_when_year_set(self):
        f = QueryFilters(min_year=2020)
        assert f.is_empty() is False

    def test_not_empty_when_citations_set(self):
        f = QueryFilters(has_citations=True)
        assert f.is_empty() is False

    def test_to_dict_keys(self):
        f = QueryFilters(source_type="blog", min_year=2021)
        d = f.to_dict()
        assert set(d.keys()) == {"source_type", "min_year", "max_year", "has_citations"}


class TestExtractFiltersSourceType:
    def test_research_paper_keyword(self):
        f = extract_filters("show me research papers on transformers")
        assert f.source_type == "research_paper"

    def test_academic_paper_keyword(self):
        f = extract_filters("find academic papers about LLMs")
        assert f.source_type == "research_paper"

    def test_blog_keyword(self):
        f = extract_filters("blog posts about fine-tuning")
        assert f.source_type == "blog"

    def test_documentation_keyword(self):
        f = extract_filters("show documentation for pytorch")
        assert f.source_type == "documentation"

    def test_forum_keyword(self):
        f = extract_filters("forum discussions about RAG")
        assert f.source_type == "forum"

    def test_no_source_keyword_returns_none(self):
        f = extract_filters("what is the best vector database")
        assert f.source_type is None

    def test_case_insensitive(self):
        f = extract_filters("RESEARCH PAPERS on embeddings")
        assert f.source_type == "research_paper"


class TestExtractFiltersYears:
    def test_after_year(self):
        f = extract_filters("papers after 2020 on BERT")
        assert f.min_year == 2020
        assert f.max_year is None

    def test_since_year(self):
        f = extract_filters("papers since 2019")
        assert f.min_year == 2019

    def test_before_year(self):
        f = extract_filters("articles before 2022")
        assert f.max_year == 2022
        assert f.min_year is None

    def test_in_exact_year(self):
        f = extract_filters("papers in 2021 about attention")
        assert f.min_year == 2021
        assert f.max_year == 2021

    def test_year_range(self):
        f = extract_filters("research from 2018-2023")
        assert f.min_year == 2018
        assert f.max_year == 2023

    def test_recent_sets_min_year(self):
        f = extract_filters("recent papers on diffusion models")
        assert f.min_year is not None
        assert f.min_year >= 2020

    def test_latest_sets_min_year(self):
        f = extract_filters("latest research on GPT")
        assert f.min_year is not None

    def test_no_year_returns_none(self):
        f = extract_filters("how does RAG work")
        assert f.min_year is None
        assert f.max_year is None


class TestExtractFiltersCitations:
    def test_with_citations_keyword(self):
        f = extract_filters("papers with citations on NLP")
        assert f.has_citations is True

    def test_highly_cited_keyword(self):
        f = extract_filters("highly cited papers about attention")
        assert f.has_citations is True

    def test_peer_reviewed_keyword(self):
        f = extract_filters("peer reviewed articles on ML")
        assert f.has_citations is True

    def test_no_citation_keyword(self):
        f = extract_filters("papers on vector search")
        assert f.has_citations is None


class TestExtractFiltersCombined:
    def test_source_and_year_combined(self):
        f = extract_filters("research papers after 2021 with citations")
        assert f.source_type == "research_paper"
        assert f.min_year == 2021
        assert f.has_citations is True

    def test_empty_query_returns_empty_filters(self):
        f = extract_filters("what is machine learning")
        assert f.is_empty() is True


class TestNormalizeQuery:
    def test_polar_to_polars(self):
        result = normalize_query("how to use polar dataframes")
        assert "polars" in result

    def test_scikit_learn_normalised(self):
        result = normalize_query("scikit learn classification")
        assert "scikit-learn" in result

    def test_tf_to_tensorflow(self):
        result = normalize_query("train a model in tf")
        assert "tensorflow" in result

    def test_no_match_unchanged(self):
        query = "how does attention mechanism work"
        assert normalize_query(query) == query

    def test_case_insensitive_replacement(self):
        result = normalize_query("using TF for image classification")
        assert "tensorflow" in result.lower()

    def test_pyspark_unchanged(self):
        result = normalize_query("pyspark dataframe operations")
        assert "pyspark" in result

    def test_pt_to_pytorch(self):
        result = normalize_query("train model with pt")
        assert "pytorch" in result
