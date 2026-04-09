"""Tests for the retriever module (Document, TextSplitter, BM25Retriever)."""

import pytest

from toy_agent.retriever import BM25Retriever, Document, TextSplitter


class TestTextSplitter:
    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size is returned as one chunk."""
        splitter = TextSplitter(chunk_size=100, overlap=10)
        chunks = splitter.split("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text_returns_empty(self):
        """Empty or whitespace-only text returns no chunks."""
        splitter = TextSplitter()
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_long_text_splits_into_chunks(self):
        """Long text is split into multiple chunks."""
        splitter = TextSplitter(chunk_size=50, overlap=10)
        text = "word " * 100  # ~500 chars
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # All content should be preserved (approximately)
        assert len("".join(chunks)) > 400

    def test_paragraph_break_preferred(self):
        """Splitter prefers paragraph breaks over mid-sentence cuts."""
        splitter = TextSplitter(chunk_size=60, overlap=10)
        text = "First paragraph with some text.\n\nSecond paragraph with more text.\n\nThird paragraph."
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # Chunks should not break mid-paragraph when possible
        for chunk in chunks:
            assert len(chunk) > 0


class TestBM25Retriever:
    def test_index_and_query(self):
        """Indexed documents can be queried and relevant results are returned."""
        retriever = BM25Retriever()
        retriever.index(
            [
                Document(content="Python is a programming language", metadata={"source": "lang"}),
                Document(content="Rust is a systems programming language", metadata={"source": "lang"}),
                Document(content="The cat sat on the mat", metadata={"source": "story"}),
            ]
        )

        results = retriever.query("programming language", top_k=2)
        assert len(results) == 2
        # The programming-related docs should rank higher
        assert all("programming" in r.content for r in results)

    def test_empty_query_returns_empty(self):
        """Querying an empty index returns no results."""
        retriever = BM25Retriever()
        results = retriever.query("anything", top_k=3)
        assert results == []

    def test_multiple_index_calls_accumulate(self):
        """Calling index() multiple times accumulates documents."""
        retriever = BM25Retriever()
        retriever.index([Document(content="First document about cats")])
        retriever.index([Document(content="Second document about dogs")])

        results = retriever.query("cats", top_k=1)
        assert len(results) == 1
        assert "cats" in results[0].content

    def test_top_k_limits_results(self):
        """top_k parameter limits the number of returned results."""
        retriever = BM25Retriever()
        retriever.index(
            [
                Document(content="Apple fruit"),
                Document(content="Banana fruit"),
                Document(content="Cherry fruit"),
                Document(content="Date fruit"),
            ]
        )

        results = retriever.query("fruit", top_k=2)
        assert len(results) == 2

    def test_metadata_preserved(self):
        """Metadata from documents is preserved in query results."""
        retriever = BM25Retriever()
        retriever.index([Document(content="Test content", metadata={"source": "test.txt", "author": "Alice"})])

        results = retriever.query("test", top_k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "test.txt"
        assert results[0].metadata["author"] == "Alice"

    def test_long_document_is_chunked(self):
        """Long documents are split into chunks by the splitter."""
        retriever = BM25Retriever(splitter=TextSplitter(chunk_size=50, overlap=10))
        retriever.index([Document(content="word " * 200, metadata={"source": "long"})])

        # Should have multiple chunks
        assert len(retriever._chunks) > 1


class TestAgentRetrieverIntegration:
    @pytest.mark.anyio
    async def test_retriever_injects_context(self):
        """Agent with retriever injects relevant context into messages."""
        from unittest.mock import MagicMock

        from toy_agent.agent import Agent

        retriever = BM25Retriever()
        retriever.index([Document(content="Python is a programming language")])

        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "Python is a programming language"
        client.chat.completions.create.return_value = response

        agent = Agent(client=client, retriever=retriever)
        await agent.run("What is Python?")

        # Verify the RAG context message was injected before the LLM call
        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        rag_messages = [m for m in messages if m.get("content", "").startswith("[Retrieved context]")]
        assert len(rag_messages) == 1
        assert "Python" in rag_messages[0]["content"]

    @pytest.mark.anyio
    async def test_no_retriever_no_injection(self):
        """Agent without retriever works normally (no context injection)."""
        from unittest.mock import MagicMock

        from toy_agent.agent import Agent

        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "Hello"
        client.chat.completions.create.return_value = response

        agent = Agent(client=client)
        await agent.run("hi")

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        rag_messages = [m for m in messages if m.get("content", "").startswith("[Retrieved context]")]
        assert len(rag_messages) == 0
