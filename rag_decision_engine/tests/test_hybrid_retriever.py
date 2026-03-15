import pytest
from rag_decision_engine.retrieval.bm25_retriever import BM25Retriever
from rag_decision_engine.retrieval.hybrid_retriever import HybridRetriever
CORPUS = [
    {
        "id": "c1",
        "text": "XGBoost is a gradient boosting framework optimised for tabular data.",
        "meta": {"doc_id": "d1", "source_type": "technical_blog"},
    },
    {
        "id": "c2",
        "text": "Random Forest is an ensemble method using bagging of decision trees.",
        "meta": {"doc_id": "d2", "source_type": "research_paper"},
    },
    {
        "id": "c3",
        "text": "Neural networks excel at image and text tasks but struggle on small tabular data.",
        "meta": {"doc_id": "d3", "source_type": "technical_blog"},
    },
    {
        "id": "c4",
        "text": "XGBoost often wins Kaggle competitions involving structured/tabular datasets.",
        "meta": {"doc_id": "d4", "source_type": "forum"},
    },
]
class TestBM25Retriever:
    def setup_method(self):
        self.retriever = BM25Retriever()
        for doc in CORPUS:
            self.retriever.add_documents([doc["text"]], [doc["id"]], [doc["meta"]])
    def test_returns_results(self):
        results = self.retriever.search("XGBoost tabular data")
        assert len(results) > 0
    def test_relevant_first(self):
        results = self.retriever.search("XGBoost tabular data", top_k=4)
        top_ids = [r.chunk_id for r in results[:2]]
        assert "c1" in top_ids or "c4" in top_ids
    def test_empty_corpus_returns_empty(self):
        r = BM25Retriever()
        assert r.search("anything") == []
    def test_scores_are_normalised(self):
        results = self.retriever.search("XGBoost")
        for r in results:
            assert 0.0 <= r.score <= 1.0
class TestHybridRetriever:
    def test_parse_options(self):
        from rag_decision_engine.services.decision_service import parse_options
        opts = parse_options("Should I use XGBoost or Random Forest?")
        assert len(opts) == 2
    def test_parse_options_vs(self):
        from rag_decision_engine.services.decision_service import parse_options
        opts = parse_options("PyTorch vs TensorFlow")
        assert len(opts) == 2
    def test_parse_single_option(self):
        from rag_decision_engine.services.decision_service import parse_options
        opts = parse_options("What is the best ML framework?")
        assert len(opts) == 1
