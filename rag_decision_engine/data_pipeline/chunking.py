from dataclasses import dataclass, field
from typing import Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_decision_engine.config import settings
@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict[str, object] = field(default_factory=dict)
def chunk_document(
    doc_id: str,
    text: str,
    metadata: dict[str, object] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks: list[str] = splitter.split_text(text)
    meta = metadata or {}
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}_{i}",
            doc_id=doc_id,
            text=chunk,
            chunk_index=i,
            metadata={**meta, "chunk_index": i, "total_chunks": len(raw_chunks)},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
def iter_chunks(
    doc_id: str,
    text: str,
    metadata: dict[str, object] | None = None,
) -> Iterator[DocumentChunk]:
    yield from chunk_document(doc_id, text, metadata)
