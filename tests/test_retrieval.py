"""
tests/test_retrieval.py
Unit tests for the retrieval layer (vector store, hybrid search, reranker).

This file will grow as we add re-ranking in Step 7.
"""

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchEngine


# =============================================================================
# Vector Store Tests
# =============================================================================
class TestVectorStore:
    """Tests for the ChromaDB wrapper."""

    @pytest.fixture
    def temp_vector_store(self, tmp_path: Path) -> VectorStore:
        """Fresh vector store for each test, using a temporary directory."""
        store = VectorStore(
            collection_name="test_docs",
            persist_directory=str(tmp_path / "vector_store"),
        )
        store.reset()  # Clean slate
        return store

    def test_add_and_query(self, temp_vector_store: VectorStore) -> None:
        """Documents added must be retrievable by semantic query."""
        texts = ["hello world", "machine learning", "deep learning"]
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar to query [1,0,0]
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        metadatas = [{"type": "txt"}, {"type": "pdf"}, {"type": "txt"}]

        temp_vector_store.add_documents(texts, embeddings, metadatas)

        results = temp_vector_store.query(embedding=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # The closest vector to [1,0,0] should be [1,0,0] itself
        assert results[0]["text"] == "hello world"
        assert results[0]["metadata"]["type"] == "txt"

    def test_metadata_filter(self, temp_vector_store: VectorStore) -> None:
        """Query with 'where' filter must only return matching metadata."""
        texts = ["doc a", "doc b"]
        embeddings = [[1.0, 0.0], [0.9, 0.1]]
        metadatas = [{"type": "pdf"}, {"type": "txt"}]

        temp_vector_store.add_documents(texts, embeddings, metadatas)

        # Filter for PDF only
        results = temp_vector_store.query(
            embedding=[1.0, 0.0], top_k=10, where={"type": "pdf"}
        )
        assert len(results) == 1
        assert results[0]["metadata"]["type"] == "pdf"

    def test_empty_store_query(self, temp_vector_store: VectorStore) -> None:
        """Querying an empty store must return empty results, not crash."""
        results = temp_vector_store.query(embedding=[1.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_add_empty_texts_is_noop(self, temp_vector_store: VectorStore) -> None:
        """Adding empty texts must be a no-op and log a warning."""
        temp_vector_store.add_documents([], [], [])
        assert temp_vector_store.count() == 0

    def test_batch_insertion(self, temp_vector_store: VectorStore) -> None:
        """Adding many documents in small batches must work correctly."""
        n = 250
        texts = [f"document {i}" for i in range(n)]
        embeddings = [[1.0, 0.0] for _ in range(n)]
        metadatas = [{"idx": str(i)} for i in range(n)]

        temp_vector_store.add_documents(texts, embeddings, metadatas, batch_size=50)
        assert temp_vector_store.count() == n

    def test_get_by_id(self, temp_vector_store: VectorStore) -> None:
        """Retrieving a document by ID must return correct text and metadata."""
        temp_vector_store.add_documents(
            texts=["target doc"],
            embeddings=[[1.0, 0.0]],
            metadatas={"key": "value"},
            ids=["doc_42"],
        )

        result = temp_vector_store.get_by_id("doc_42")
        assert result is not None
        assert result["text"] == "target doc"
        assert result["metadata"]["key"] == "value"

    def test_get_by_missing_id_returns_none(self, temp_vector_store: VectorStore) -> None:
        """Querying a non-existent ID must return None gracefully."""
        result = temp_vector_store.get_by_id("nonexistent")
        assert result is None

    def test_reset_clears_all(self, temp_vector_store: VectorStore) -> None:
        """Reset must delete all documents and return count to zero."""
        temp_vector_store.add_documents(
            texts=["temp"],
            embeddings=[[1.0, 0.0]],
            metadatas={"type": "tmp"},
        )
        assert temp_vector_store.count() == 1

        temp_vector_store.reset()
        assert temp_vector_store.count() == 0


# =============================================================================
# Hybrid Search Tests
# =============================================================================
class TestHybridSearchEngine:
    """Tests for dense + sparse fusion via RRF."""

    @pytest.fixture
    def hybrid_engine(self, tmp_path: Path) -> HybridSearchEngine:
        """Fresh hybrid engine with an empty but initialized vector store."""
        store = VectorStore(
            collection_name="hybrid_test",
            persist_directory=str(tmp_path / "vector_store"),
        )
        store.reset()
        return HybridSearchEngine(vector_store=store)

    def test_sparse_without_index_returns_empty(self, hybrid_engine: HybridSearchEngine) -> None:
        """BM25 search before indexing must return empty list gracefully."""
        results = hybrid_engine.search_sparse("test query")
        assert results == []

    def test_rrf_boosts_documents_in_both_lists(self) -> None:
        """
        If a document appears in both dense and sparse results, its RRF score
        should be higher than a document that appears in only one list.
        """
        # Mock two result lists with overlapping doc "shared"
        dense = [
            {"id": "shared", "text": "shared doc", "metadata": {}, "rank": 1, "source": "dense"},
            {"id": "dense_only", "text": "dense doc", "metadata": {}, "rank": 2, "source": "dense"},
        ]
        sparse = [
            {"id": "shared", "text": "shared doc", "metadata": {}, "rank": 3, "source": "sparse"},
            {"id": "sparse_only", "text": "sparse doc", "metadata": {}, "rank": 1, "source": "sparse"},
        ]

        engine = HybridSearchEngine(vector_store=None)  # type: ignore
        fused = engine._reciprocal_rank_fusion(dense, sparse)

        # "shared" should be first because it appears in both lists
        assert fused[0]["id"] == "shared"
        assert "dense" in fused[0]["sources"]
        assert "sparse" in fused[0]["sources"]

    def test_rrf_preserves_single_list_documents(self) -> None:
        """
        Documents that appear in only one list should still appear in the
        fused results, just with lower scores.
        """
        dense = [
            {"id": "only_dense", "text": "d", "metadata": {}, "rank": 1, "source": "dense"},
        ]
        sparse = [
            {"id": "only_sparse", "text": "s", "metadata": {}, "rank": 1, "source": "sparse"},
        ]

        engine = HybridSearchEngine(vector_store=None)  # type: ignore
        fused = engine._reciprocal_rank_fusion(dense, sparse)

        ids = [r["id"] for r in fused]
        assert "only_dense" in ids
        assert "only_sparse" in ids

    def test_end_to_end_hybrid_search(self, tmp_path: Path) -> None:
        """
        Full pipeline: ingest docs, build BM25 index, query with both
        dense and sparse signals, verify RRF fusion works.
        """
        from src.embedding.embedder import Embedder

        store = VectorStore(
            collection_name="e2e_hybrid",
            persist_directory=str(tmp_path / "vector_store"),
        )
        store.reset()

        # Corpus: 3 documents
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "RAG stands for Retrieval Augmented Generation in NLP",
        ]
        metadatas = [{"idx": "0"}, {"idx": "1"}, {"idx": "2"}]
        ids = ["doc_0", "doc_1", "doc_2"]

        # Embed and store
        embedder = Embedder()
        embeddings = embedder.embed(texts)

        store.add_documents(texts, embeddings.tolist(), metadatas, ids=ids)

        # Build hybrid engine
        engine = HybridSearchEngine(vector_store=store)
        engine.build_bm25_index(texts, metadatas, ids)

        # Query: "RAG pipeline" — should match doc_2 via dense (semantic)
        # and also potentially via sparse (keyword "RAG")
        query = "RAG pipeline"
        query_emb = embedder.embed_single(query).tolist()

        results = engine.search_hybrid(query=query, query_embedding=query_emb)

        assert len(results) > 0
        # The RAG document should be highly ranked
        top_ids = [r["id"] for r in results]
        assert "doc_2" in top_ids

        # Verify RRF scores are present
        assert all("rrf_score" in r for r in results)
        assert all("sources" in r for r in results)