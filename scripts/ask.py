"""
scripts/ask.py
Complete end-to-end RAG CLI: question → answer with citations.

Usage:
    python scripts/ask.py "What is RAG?"
"""

import argparse

from config.log_config import configure_logging
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.rag_chain import RAGChain


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Ask a question to the RAG engine")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    args = parser.parse_args()

    # Initialize components
    embedder = Embedder()
    store = VectorStore()
    engine = HybridSearchEngine(vector_store=store)
    reranker = None if args.no_rerank else CrossEncoderReranker()
    generator = RAGChain()

    # Build BM25 index from stored documents if available
    count = store.count()
    if count > 0:
        # For the CLI, we skip full BM25 rebuild; dense-only fallback
        pass

    # Embed query
    query_emb = embedder.embed_single(args.question).tolist()

    # Retrieve
    results = engine.search_hybrid(
        query=args.question,
        query_embedding=query_emb,
    )

    # Re-rank
    if reranker and results:
        results = reranker.rerank(args.question, results, top_k=args.top_k)

    # Generate answer
    response = generator.generate(args.question, results)

    # Output
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")
    print(response["answer"])
    print(f"\n{'='*60}")
    print("Sources:")
    for s in response["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']})")
    print(f"\nLatency: {response['latency_ms']}ms | Tokens: {response['token_usage']}")
    if response["error"]:
        print(f"Error: {response['error']}")


if __name__ == "__main__":
    main()