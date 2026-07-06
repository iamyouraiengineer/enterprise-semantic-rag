

import os
from typing import List

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import get_settings


class Embedder:
    """
    Production-grade embedding engine.
    Converts text chunks into dense, normalized vector representations.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.
            device: torch device string. 'auto' triggers CUDA > MPS > CPU.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = self._resolve_device(device or settings.embedding_device)

        logger.info(
            "Loading embedding model | model={} | device={}",
            self.model_name,
            self.device,
        )

        # Load model. SentenceTransformer auto-downloads to ~/.cache/torch/...
        self._model = SentenceTransformer(self.model_name, device=self.device)

        # Cache the embedding dimension for downstream validation
        self.embedding_dim = self._model.get_embedding_dimension()
        logger.info(
            "Embedding model ready | dim={} | model={}",
            self.embedding_dim,
            self.model_name,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into a dense embedding matrix.

        Args:
            texts: List of strings to embed. Empty strings are valid but
                   produce near-zero vectors.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
            L2-normalized.
        """
        if not texts:
            logger.warning("embed() called with empty text list")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        logger.debug("Embedding batch | count={} | model={}", len(texts), self.model_name)

        # SentenceTransformer.encode handles batching internally, but we
        # expose batch_size as a tunable parameter for production tuning.
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize: cosine = dot product
        )

        # Ensure correct dtype and shape
        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.shape == (len(texts), self.embedding_dim), (
            f"Shape mismatch: expected {(len(texts), self.embedding_dim)}, "
            f"got {embeddings.shape}"
        )

        logger.debug(
            "Embedding complete | shape={} | norm_mean={:.4f}",
            embeddings.shape,
            float(np.mean(np.linalg.norm(embeddings, axis=1))),
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience wrapper for embedding a single string.
        Returns a 1-D array of shape (embedding_dim,).
        """
        result = self.embed([text])
        return result[0]

    def _resolve_device(self, device: str) -> str:
        """
        Resolve 'auto' to the best available hardware accelerator.
        Priority: CUDA > MPS (Apple Silicon) > CPU.
        """
        if device != "auto":
            return device

        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU acceleration.")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected. Using Metal acceleration.")
            return "mps"
        else:
            logger.info("No GPU detected. Falling back to CPU.")
            return "cpu"