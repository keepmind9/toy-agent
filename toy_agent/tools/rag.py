"""RAG tools: index documents and search for relevant information."""

from toy_agent.retriever import BM25Retriever, Document
from toy_agent.tools import tool

# Module-level singleton — shared across all calls within the same agent session
_retriever = BM25Retriever()


@tool(description="Index a text document for later retrieval via rag_search")
def rag_index(content: str, source: str = "") -> str:
    """content: The text content to index
    source: Optional source identifier (e.g. filename, URL)"""
    try:
        _retriever.index([Document(content=content, metadata={"source": source})])
        return f"OK: indexed {len(content)} characters"
    except Exception as e:
        return f"Error: {e}"


@tool(description="Search indexed documents for information relevant to a query")
def rag_search(query: str, top_k: int = 3) -> str:
    """query: The search query
    top_k: Number of results to return (default 3)"""
    try:
        results = _retriever.query(query, top_k=top_k)
        if not results:
            return "No results found"
        return "\n---\n".join(f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.content}" for r in results)
    except Exception as e:
        return f"Error: {e}"
