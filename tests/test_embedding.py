"""
tests/test_embedding.py
Unit tests for the embedding engine.

Design decisions:
- Tests verify output shape, dtype, normalization, and device logic.
- We use a small model (all-MiniLM-L6-v2) to keep test execution fast.
- We do NOT test CUDA-specific behavior because CI may not have GPUs.
- The singleton conftest.py resets settings before each test.
"""

import numpy as np
import pytest

from src.embedding.embedder import Embedder


class TestEmbedder:
    """Comprehensive tests for the Embedder class."""

    @pytest.fixture(scope="class")
    def embedder(self) -> Embedder:
        """
        Class-scoped fixture: load model once for all tests in this class.
        This avoids re-downloading/re-loading the model for every test method.
        """
        return Embedder()

    def test_embedding_shape(self, embedder: Embedder) -> None:
        """A batch of 3 texts must return shape (3, embedding_dim)."""
        texts = ["hello world", "machine learning", "RAG pipeline"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape == (3, embedder.embedding_dim)

    def test_single_embedding_shape(self, embedder: Embedder) -> None:
        """embed_single must return a 1-D array of length embedding_dim."""
        embedding = embedder.embed_single("test")
        assert embedding.shape == (embedder.embedding_dim,)
        assert embedding.ndim == 1

    def test_embedding_dtype(self, embedder: Embedder) -> None:
        """Embeddings must be float32 for memory efficiency and ChromaDB compatibility."""
        embeddings = embedder.embed(["test"])
        assert embeddings.dtype == np.float32

    def test_empty_list_returns_zero_shape(self, embedder: Embedder) -> None:
        """Empty input must return a zero-row array with correct embedding_dim."""
        embeddings = embedder.embed([])
        assert embeddings.shape == (0, embedder.embedding_dim)

    def test_embeddings_are_normalized(self, embedder: Embedder) -> None:
        """
        L2 norms must be approximately 1.0 for all embeddings.
        This verifies that normalize_embeddings=True is working.
        """
        texts = ["first sentence", "second sentence", "third sentence"]
        embeddings = embedder.embed(texts)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_semantic_similarity(self, embedder: Embedder) -> None:
        """
        Cosine similarity between semantically related texts should be higher
        than between unrelated texts. This is an end-to-end sanity check.
        """
        embeddings = embedder.embed([
            "The cat sat on the mat",
            "A feline rested on the rug",
            "Quantum computing is complex",
        ])

        # Compute cosine similarities (dot product since normalized)
        sim_cat_feline = float(np.dot(embeddings[0], embeddings[1]))
        sim_cat_quantum = float(np.dot(embeddings[0], embeddings[2]))

        assert sim_cat_feline > sim_cat_quantum, (
            f"Related texts should be more similar: "
            f"cat-feline={sim_cat_feline:.4f} vs cat-quantum={sim_cat_quantum:.4f}"
        )

    def test_embedding_dimension_matches_model(self, embedder: Embedder) -> None:
        """all-MiniLM-L6-v2 produces 384-dimensional embeddings."""
        assert embedder.embedding_dim == 384

    def test_device_resolution_cpu(self) -> None:
        """When device='cpu' is forced, _resolve_device returns 'cpu'."""
        e = Embedder(device="cpu")
        assert e.device == "cpu"