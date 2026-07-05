"""
scripts/test_embed.py
Quick manual test of the embedding engine.

Usage:
    python scripts/test_embed.py
"""

from config.log_config import configure_logging
from src.embedding.embedder import Embedder


def main() -> None:
    configure_logging()

    embedder = Embedder()
    print(f"Model: {embedder.model_name}")
    print(f"Device: {embedder.device}")
    print(f"Embedding dimension: {embedder.embedding_dim}")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above a sleepy hound.",
        "Machine learning transforms data into predictions.",
    ]

    embeddings = embedder.embed(texts)
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Cosine similarity (dot product since normalized)
    import numpy as np
    sim_0_1 = float(np.dot(embeddings[0], embeddings[1]))
    sim_0_2 = float(np.dot(embeddings[0], embeddings[2]))
    print(f"\nSimilarity (fox vs canine): {sim_0_1:.4f}")
    print(f"Similarity (fox vs ML):     {sim_0_2:.4f}")


if __name__ == "__main__":
    main()