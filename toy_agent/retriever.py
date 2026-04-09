"""Retriever module for RAG (Retrieval-Augmented Generation).

Provides:
- Document: data class for text + metadata
- TextSplitter: fixed-size chunking with overlap
- BaseRetriever: abstract interface (swap BM25 for embedding-based later)
- BM25Retriever: keyword-based retrieval using rank_bm25
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """A chunk of text with optional metadata."""

    content: str
    metadata: dict = field(default_factory=dict)


class TextSplitter:
    """Split text into fixed-size chunks with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        """Split text into chunks of roughly chunk_size characters.

        Tries to break on paragraph boundaries (double newlines) when possible.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break on paragraph boundary within the last `overlap` chars
            if end < len(text):
                search_start = max(start + self.chunk_size - self.overlap, start)
                para_break = text.find("\n\n", search_start, end + self.overlap)
                if para_break != -1:
                    end = para_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end if end == len(text) else end - self.overlap

        return chunks


class BaseRetriever(ABC):
    """Abstract retriever interface. Implementations can use BM25, embeddings, etc."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Add documents to the index."""

    @abstractmethod
    def query(self, question: str, top_k: int = 3) -> list[Document]:
        """Return the top_k most relevant documents for the question."""


class BM25Retriever(BaseRetriever):
    """BM25-based retriever using the rank_bm25 library."""

    def __init__(self, splitter: TextSplitter | None = None):
        self.splitter = splitter or TextSplitter()
        self._chunks: list[Document] = []
        self._bm25 = None

    def index(self, documents: list[Document]) -> None:
        """Split documents into chunks and (re)build the BM25 index."""
        for doc in documents:
            text_chunks = self.splitter.split(doc.content)
            for i, chunk in enumerate(text_chunks):
                self._chunks.append(
                    Document(
                        content=chunk,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        self._build_index()

    def query(self, question: str, top_k: int = 3) -> list[Document]:
        """Return the top_k most relevant chunks for the query."""
        if not self._chunks or self._bm25 is None:
            return []

        tokenized_query = question.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._chunks[idx] for idx, _ in ranked[:top_k]]

    def _build_index(self) -> None:
        """Build or rebuild the BM25 index from current chunks."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [chunk.content.lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
