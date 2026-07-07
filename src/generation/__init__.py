"""
src/generation package
RAG generation chain and LLM integration.
"""

from src.generation.rag_chain import RAGChain, RAGGenerationError

__all__ = ["RAGChain", "RAGGenerationError"]