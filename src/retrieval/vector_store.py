
import os
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config.settings import get_settings


class VectorStore:
    """
    Production-grade wrapper around ChromaDB PersistentClient.
    Handles collection lifecycle, batch insertion, and metadata-aware querying.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(settings.vector_store_path)

        # Ensure the directory exists before ChromaDB tries to write to it
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(
            "Initializing VectorStore | collection={} | persist_dir={}",
            self.collection_name,
            self.persist_directory,
        )

        # ChromaDB PersistentClient writes to disk automatically
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable phone-home
            ),
        )

        self._collection: Optional[chromadb.Collection] = None

    # ------------------------------------------------------------------
    # Collection Lifecycle
    # ------------------------------------------------------------------
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Lazy initialization: create collection on first use."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # We L2-normalized, so cosine = dot
            )
            logger.info(
                "Collection ready | name={} | count={}",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    def reset(self) -> None:
        """
        Delete and recreate the collection. Useful for testing and re-ingestion.
        WARNING: This permanently destroys all stored vectors.
        """
        logger.warning("Resetting collection | name={}", self.collection_name)
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection may not exist yet
        self._collection = None
        self._get_or_create_collection()

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self._get_or_create_collection().count()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Union[Dict, List[Dict]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Add a batch of documents with their embeddings and metadata.

        Args:
            texts: The raw text of each chunk.
            embeddings: Pre-computed dense vectors (must match collection dimension).
            metadatas: Per-chunk metadata dicts, or a single dict for uniform metadata.
            ids: Optional custom IDs. If None, auto-generated as index strings.
            batch_size: Insert in chunks to respect ChromaDB limits.
        """
        if not texts:
            logger.warning("add_documents called with empty texts")
            return

        n = len(texts)
        if len(embeddings) != n:
            raise ValueError(
                f"Mismatch: texts={n}, embeddings={len(embeddings)}"
            )

        # Normalize metadatas: single dict -> list of n copies; list -> verify length
        if isinstance(metadatas, dict):
            metadatas_list: List[Dict] = [metadatas] * n
        else:
            metadatas_list = metadatas
            if len(metadatas_list) != n:
                raise ValueError(
                    f"Mismatch: texts={n}, metadatas={len(metadatas_list)}"
                )

        collection = self._get_or_create_collection()
        ids = ids or [str(i) for i in range(n)]

        # Batch insertion with progress logging
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas_list[i:batch_end],
            )
            logger.debug(
                "Inserted batch | start={} | end={} | total={}",
                i,
                batch_end,
                n,
            )

        logger.info("Insertion complete | total_documents={}", n)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic similarity search by embedding vector.

        Args:
            embedding: The query vector (must be same dimension as stored vectors).
            top_k: Number of results to return.
            where: ChromaDB metadata filter dict, e.g. {"type": "pdf"}.

        Returns:
            List of result dicts with keys: id, text, metadata, distance.
        """
        collection = self._get_or_create_collection()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists: [[doc1, doc2], [meta1, meta2], [[dist1, dist2]]]
        # We flatten into a clean list of dicts.
        output: List[Dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            output.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": float(distances[i]) if i < len(distances) else 0.0,
                }
            )

        logger.debug(
            "Query executed | top_k={} | results={} | filter={}",
            top_k,
            len(output),
            where,
        )
        return output

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a single document by its ID.
        """
        collection = self._get_or_create_collection()
        try:
            result = collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
            if result and result.get("documents") and result["documents"]:
                return {
                    "id": doc_id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
        except Exception as e:
            logger.error("Failed to get document by ID | id={} | error={}", doc_id, e)
        return None