"""
tests/test_api.py
Integration tests for the FastAPI application.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Fresh TestClient with mock services injected into app.state."""
    app = create_app()

    # Create mocks
    mock_embedder = MagicMock()
    mock_embedder.model_name = "test-embedder"
    mock_embedder.embed_single.return_value = np.array([0.1, 0.2, 0.3])

    mock_store = MagicMock()
    mock_store.count.return_value = 42

    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [
        {"id": "1", "text": "test chunk", "metadata": {"source": "test.txt"}, "rrf_score": 0.9}
    ]
    mock_engine.search_dense.return_value = [
        {"id": "1", "text": "test chunk", "metadata": {"source": "test.txt"}, "distance": 0.1}
    ]
    mock_engine.search_sparse.return_value = [
        {"id": "1", "text": "test chunk", "metadata": {"source": "test.txt"}, "score": 1.5}
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"id": "1", "text": "test chunk", "metadata": {"source": "test.txt"}, "rerank_score": 0.95}
    ]

    mock_rag = MagicMock()
    mock_rag.model_name = "test-llm"
    mock_rag.generate.return_value = {
        "answer": "Test answer",
        "sources": [{"source": "test.txt", "chunk_index": 0, "text_preview": "test"}],
        "latency_ms": 100.0,
        "token_usage": 50,
        "error": None,
    }

    # Inject into app.state (bypass lifespan)
    app.state.embedder = mock_embedder
    app.state.vector_store = mock_store
    app.state.hybrid_engine = mock_engine
    app.state.reranker = mock_reranker
    app.state.rag_chain = mock_rag

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["vector_store_count"] == 42
        assert data["embedding_model"] == "test-embedder"
        assert data["llm_model"] == "test-llm"


class TestIngestEndpoint:
    def test_ingest_valid_documents(self, client: TestClient) -> None:
        payload = {
            "documents": [
                {"text": "Hello world", "source": "test.txt", "metadata": {"type": "txt"}},
                {"text": "Machine learning is AI", "source": "test2.txt"},
            ]
        }
        response = client.post("/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_ingested"] == 2

    def test_ingest_empty_documents_rejected(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"documents": []})
        assert response.status_code == 422


class TestQueryEndpoint:
    def test_query_returns_answer(self, client: TestClient) -> None:
        payload = {"question": "What is AI?", "top_k": 3, "rerank": True}
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer"
        assert "sources" in data
        assert data["retrieved_chunks"] > 0

    def test_query_missing_question_rejected(self, client: TestClient) -> None:
        response = client.post("/query", json={"top_k": 3})
        assert response.status_code == 422


class TestSearchEndpoint:
    def test_search_returns_results(self, client: TestClient) -> None:
        payload = {"query": "machine learning", "mode": "hybrid", "top_k": 5}
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["mode"] == "hybrid"
        assert len(data["results"]) > 0