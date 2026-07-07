"""
tests/test_generation.py
Unit tests for the RAG generation chain.

Design decisions:
- We mock the HTTP client to avoid real LLM API calls in tests.
- We test prompt assembly, context truncation, source extraction, and
  graceful error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.rag_chain import RAGChain, RAGGenerationError


class TestRAGChain:
    """Tests for the RAG generation chain."""

    @pytest.fixture
    def rag_chain(self) -> RAGChain:
        """Initialize RAGChain with mocked settings."""
        with patch("src.generation.rag_chain.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openai",
                llm_model_name="gpt-4o-mini",
                openai_api_key="test-key",
                openai_base_url="https://api.openai.com/v1",
                llm_temperature=0.1,
                llm_max_tokens=1024,
                ollama_base_url="http://localhost:11434",
            )
            return RAGChain()

    def test_generate_with_empty_documents(self, rag_chain: RAGChain) -> None:
        """Empty documents must return the 'cannot answer' fallback."""
        result = rag_chain.generate("What is AI?", [])
        assert "cannot answer" in result["answer"].lower()
        assert result["sources"] == []
        assert result["error"] is None

    def test_format_context(self, rag_chain: RAGChain) -> None:
        """Context blocks must include source path and chunk index."""
        docs = [
            {
                "text": "Machine learning is a subset of AI.",
                "metadata": {"source": "doc1.txt", "chunk_index": 0},
            }
        ]
        blocks = rag_chain._format_context(docs)
        assert len(blocks) == 1
        assert "doc1.txt" in blocks[0]
        assert "chunk 0" in blocks[0]
        assert "Machine learning" in blocks[0]

    def test_truncate_context_preserves_top_chunks(self, rag_chain: RAGChain) -> None:
        """Truncation must keep the highest-ranked chunks at the top."""
        # Create blocks that would exceed the token budget
        huge_blocks = [f"[{i}] Source: doc.txt (chunk {i})\n{'x' * 1000}" for i in range(50)]
        result = rag_chain._truncate_context(huge_blocks)
        # Should be truncated but not empty
        assert len(result) > 0
        assert "[1]" in result  # First chunk preserved

    def test_extract_sources_deduplicates(self, rag_chain: RAGChain) -> None:
        """Duplicate source+chunk_index pairs must be deduplicated."""
        docs = [
            {"text": "a", "metadata": {"source": "s1.txt", "chunk_index": 0}},
            {"text": "b", "metadata": {"source": "s1.txt", "chunk_index": 0}},  # duplicate
            {"text": "c", "metadata": {"source": "s2.txt", "chunk_index": 1}},
        ]
        sources = rag_chain._extract_sources(docs)
        assert len(sources) == 2
        sources_list = [(s["source"], s["chunk_index"]) for s in sources]
        assert ("s1.txt", 0) in sources_list
        assert ("s2.txt", 1) in sources_list

    @patch("src.generation.rag_chain.httpx.Client.post")
    def test_call_openai_success(self, mock_post, rag_chain: RAGChain) -> None:
        """Successful OpenAI call must return answer and token count."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " Paris is the capital."}}],
            "usage": {"total_tokens": 42},
        }
        mock_post.return_value = mock_response

        answer, tokens = rag_chain._call_openai("test prompt")
        assert "Paris" in answer
        assert tokens == 42

    @patch("src.generation.rag_chain.httpx.Client.post")
    def test_generate_graceful_on_api_error(self, mock_post, rag_chain: RAGChain) -> None:
        """API failure must return error flag and fallback answer with context."""
        mock_post.side_effect = Exception("Connection timeout")

        docs = [{"text": "Context here", "metadata": {"source": "test.txt", "chunk_index": 0}}]
        result = rag_chain.generate("What is this?", docs)

        assert result["error"] is not None
        assert "Context here" in result["answer"]
        assert result["sources"][0]["source"] == "test.txt"

    def test_unsupported_provider_raises(self, rag_chain: RAGChain) -> None:
        """An unsupported provider must raise RAGGenerationError."""
        rag_chain.provider = "unknown_provider"
        with pytest.raises(RAGGenerationError):
            rag_chain._call_llm("test")

    def test_call_openai_without_key_raises(self, rag_chain: RAGChain) -> None:
        """Missing API key must raise RAGGenerationError before HTTP call."""
        rag_chain.api_key = ""
        with pytest.raises(RAGGenerationError):
            rag_chain._call_openai("test prompt")