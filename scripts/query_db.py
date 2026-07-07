"""
scripts/query_db.py
CLI entrypoint for querying with hybrid search + re-ranking.

Usage:
    python scripts/query_db.py "your search query"
    python scripts/query_db.py "machine learning" --mode hybrid --rerank
"""

import argparse
import json

from config.log_config import configure_logging
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Query the RAG engine")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Search mode",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder re-ranking to hybrid results",
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Metadata filter JSON"
    )
    args = parser.parse_args()

    embedder = Embedder()
    store = VectorStore()
    engine = HybridSearchEngine(vector_store=store)
    reranker = CrossEncoderReranker() if args.rerank else None

    where = None
    if args.filter:
        where = json.loads(args.filter)

    print(f"\nQuery: '{args.query}' | Mode: {args.mode} | Rerank: {args.rerank}\n")

    if args.mode == "dense":
        query_emb = embedder.embed_single(args.query).tolist()
        results = engine.search_dense(query_emb, top_k=args.top_k, where=where)

    elif args.mode == "sparse":
        results = engine.search_sparse(args.query, top_k=args.top_k)

    else:  # hybrid
        query_emb = embedder.embed_single(args.query).tolist()
        results = engine.search_hybrid(
            query=args.query,
            query_embedding=query_emb,
            where=where,
        )

    # Apply re-ranking if requested
    if reranker and results:
        results = reranker.rerank(args.query, results, top_k=args.top_k)

    print(f"Results: {len(results)}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"ID: {r['id']}")
        if "rrf_score" in r:
            print(f"RRF Score: {r['rrf_score']:.4f}")
        if "rerank_score" in r:
            print(f"Re-rank Score: {r['rerank_score']:.4f}")
        if "sources" in r:
            print(f"Sources: {r['sources']}")
        print(f"Metadata: {r['metadata']}")
        print(f"Text: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()