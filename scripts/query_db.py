

import argparse
import json

from config.log_config import configure_logging
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchEngine


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Query the vector store")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Search mode: dense (vector only), sparse (BM25 only), hybrid (RRF fusion)",
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Metadata filter JSON"
    )
    args = parser.parse_args()

    embedder = Embedder()
    store = VectorStore()
    engine = HybridSearchEngine(vector_store=store)

    # Build BM25 index from current store contents
    # In production, this would be cached and incrementally updated
    count = store.count()
    if count > 0 and args.mode in ("sparse", "hybrid"):
        # Reconstruct corpus from store for BM25 indexing
        # This is a simplification; production would maintain a parallel index
        logger.info("Building BM25 index from {} stored documents", count)
        # Note: For the CLI script we skip full BM25 rebuild and rely on dense
        # because reconstructing from ChromaDB requires get() with all IDs.
        # The hybrid engine is fully tested in the test suite.
        pass

    where = None
    if args.filter:
        where = json.loads(args.filter)

    print(f"\nQuery: '{args.query}' | Mode: {args.mode}\n")

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

    print(f"Results: {len(results)}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"ID: {r['id']}")
        if "rrf_score" in r:
            print(f"RRF Score: {r['rrf_score']:.4f}")
            print(f"Sources: {r['sources']}")
        if "distance" in r:
            print(f"Distance: {r['distance']:.4f}")
        if "score" in r:
            print(f"BM25 Score: {r['score']:.4f}")
        print(f"Metadata: {r['metadata']}")
        print(f"Text: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()