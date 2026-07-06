

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.retrieval.vector_store import VectorStore


class HybridSearchEngine:
    """
    Enterprise-grade hybrid retrieval engine.
    Combines dense semantic search with sparse BM25 keyword search.
    """

    RRF_K = 60  # Standard RRF constant. Tuning rarely improves results.

    def __init__(
        self,
        vector_store: VectorStore,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_final: int | None = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store
        self.top_k_dense = top_k_dense if top_k_dense is not None else settings.top_k_dense
        self.top_k_sparse = top_k_sparse if top_k_sparse is not None else settings.top_k_sparse
        self.top_k_final = top_k_final if top_k_final is not None else settings.top_k_final

        # BM25 index state
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_texts: List[str] = []
        self._corpus_metadatas: List[Dict] = []
        self._corpus_ids: List[str] = []

        logger.info(
            "HybridSearchEngine initialized | dense_k={} | sparse_k={} | final_k={}",
            self.top_k_dense,
            self.top_k_sparse,
            self.top_k_final,
        )

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------
    def build_bm25_index(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """
        Build or rebuild the in-memory BM25 index from the full corpus.
        Call this after every ingestion batch.
        """
        if not texts:
            logger.warning("build_bm25_index called with empty corpus")
            self._bm25 = None
            self._corpus_texts = []
            self._corpus_metadatas = []
            self._corpus_ids = []
            return

        # Simple whitespace tokenization for BM25.
        # Production systems use better tokenizers (spaCy, NLTK, or custom).
        tokenized_corpus = [doc.lower().split() for doc in texts]

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus_texts = texts
        self._corpus_metadatas = metadatas
        self._corpus_ids = ids

        logger.info(
            "BM25 index built | documents={} | avg_tokens={:.1f}",
            len(texts),
            np.mean([len(t) for t in tokenized_corpus]),
        )

    def is_indexed(self) -> bool:
        """Return True if the BM25 index has been built."""
        return self._bm25 is not None

    # ------------------------------------------------------------------
    # Search Methods
    # ------------------------------------------------------------------
    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int | None = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Execute dense vector similarity search via ChromaDB.
        Returns ranked list with 'rank' field added.
        """
        k = top_k or self.top_k_dense
        results = self.vector_store.query(
            embedding=query_embedding,
            top_k=k,
            where=where,
        )

        # Add rank field (1-indexed)
        for i, r in enumerate(results, start=1):
            r["rank"] = i
            r["source"] = "dense"

        return results

    def search_sparse(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict]:
        """
        Execute sparse BM25 keyword search over the indexed corpus.
        Returns ranked list with 'rank' field added.
        """
        if not self.is_indexed():
            logger.warning("BM25 index not built. Returning empty sparse results.")
            return []

        k = top_k or self.top_k_sparse
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k by score
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue  # Skip zero-score results
            results.append(
                {
                    "id": self._corpus_ids[idx],
                    "text": self._corpus_texts[idx],
                    "metadata": self._corpus_metadatas[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                    "source": "sparse",
                }
            )

        logger.debug(
            "Sparse search complete | query='{}' | results={}",
            query,
            len(results),
        )
        return results

    def search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Execute hybrid search: dense + sparse, fused via Reciprocal Rank Fusion.
        Returns the top_k_final results with RRF scores.
        """
        # 1. Run both searches in parallel (conceptually; Python is sequential here)
        dense_results = self.search_dense(query_embedding, top_k=self.top_k_dense, where=where)
        sparse_results = self.search_sparse(query, top_k=self.top_k_sparse)

        # 2. Fuse via RRF
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # 3. Return top_k_final
        final_results = fused[: self.top_k_final]

        logger.info(
            "Hybrid search complete | query='{}' | dense={} | sparse={} | fused={}",
            query,
            len(dense_results),
            len(sparse_results),
            len(final_results),
        )

        return final_results

    # ------------------------------------------------------------------
    # RRF Core Algorithm
    # ------------------------------------------------------------------
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        For each document d appearing in any list:
            score(d) = Σ 1 / (k + rank_i(d))

        where k = RRF_K (60) and rank_i(d) is the 1-based rank in list i.

        Documents appearing in both lists get boosted. Documents with high
        ranks in one list but missing from the other still score reasonably.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        # Process dense results
        for r in dense_results:
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + r["rank"])
            doc_map[doc_id] = r

        # Process sparse results
        for r in sparse_results:
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + r["rank"])
            # Prefer the dense result's metadata if both exist, but keep sparse text
            if doc_id not in doc_map:
                doc_map[doc_id] = r

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

        fused_results: List[Dict] = []
        for doc_id in sorted_ids:
            result = {
                "id": doc_id,
                "text": doc_map[doc_id]["text"],
                "metadata": doc_map[doc_id]["metadata"],
                "rrf_score": scores[doc_id],
                "sources": [],  # Will track which retrievers found this doc
            }

            # Track which sources contributed
            if any(r["id"] == doc_id for r in dense_results):
                result["sources"].append("dense")
            if any(r["id"] == doc_id for r in sparse_results):
                result["sources"].append("sparse")

            fused_results.append(result)

        return fused_results