import pytest
from rag_decision_engine.data_pipeline.chunking import chunk_document, DocumentChunk
class TestChunkDocument:
    SAMPLE_TEXT = " ".join(["word"] * 1000)
    def test_produces_chunks(self):
        chunks = chunk_document("doc1", self.SAMPLE_TEXT)
        assert len(chunks) > 0
    def test_chunk_ids_are_unique(self):
        chunks = chunk_document("doc1", self.SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
    def test_chunk_id_format(self):
        chunks = chunk_document("mydoc", self.SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.chunk_id.startswith("mydoc_")
    def test_metadata_propagated(self):
        meta = {"source_type": "research_paper"}
        chunks = chunk_document("d1", self.SAMPLE_TEXT, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["source_type"] == "research_paper"
    def test_chunk_index_sequential(self):
        chunks = chunk_document("d1", self.SAMPLE_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    def test_empty_text_returns_empty(self):
        chunks = chunk_document("empty", "")
        assert chunks == []
    def test_short_text_single_chunk(self):
        chunks = chunk_document("short", "hello world", chunk_size=512, chunk_overlap=0)
        assert len(chunks) == 1
    def test_custom_chunk_size(self):
        chunks = chunk_document("d1", self.SAMPLE_TEXT, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk.text) <= 200
