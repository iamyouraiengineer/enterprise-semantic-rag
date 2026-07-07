"""
src/retrieval/reranker.py
Cross-encoder re-ranking engine for precise query-document relevance scoring.

Design decisions:
- The initial retrieval (dense + sparse + RRF) is fast but approximate.
  The cross-encoder performs a second, more expensive pass on the top-K
  candidates to produce a fine-grained relevance score.
- Cross-encoders are significantly more accurate than bi-encoders for
  relevance scoring because they see the query and document jointly,
  allowing full cross-attention between tokens.
- We use a small model (ms-marco-MiniLM-L-6-v2) for speed. It is ~80MB
  and runs on CPU in <100ms per query-document pair.
- Scores are normalized via sigmoid to [0, 1] for interpretability.
- If the model fails to load, the reranker operates in pass-through mode:
  it returns the input documents unchanged with uniform scores, but still
  respects the top_k truncation parameter.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import get_settings


class CrossEncoderReranker:
    """
    Production-grade cross-encoder re-ranker.
    Scores and re-sorts retrieved documents by precise relevance.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self.batch_size = batch_size or settings.cross_encoder_batch_size
        self.device = self._resolve_device(device or settings.embedding_device)

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._pass_through = False

        try:
            logger.info(
                "Loading cross-encoder | model={} | device={}",
                self.model_name,
                self.device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()  # Disable dropout for deterministic inference
            logger.info("Cross-encoder ready")
        except Exception as e:
            logger.error(
                "Failed to load cross-encoder. Operating in pass-through mode. | error={}",
                e,
            )
            self._pass_through = True

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Re-rank a list of documents by cross-encoder relevance to the query.

        Args:
            query: The user's question.
            documents: List of document dicts from hybrid search. Each must have
                       'text' and 'id' keys.
            top_k: Return only the top N re-ranked documents. If None, return all.

        Returns:
            Documents sorted by relevance score descending, with 'rerank_score' added.
        """
        if not documents:
            logger.debug("rerank called with empty documents")
            return []

        # Pass-through mode: model failed to load, return docs with neutral scores
        # but still respect top_k truncation
        if self._pass_through:
            logger.warning("Pass-through mode: returning documents unchanged")
            for doc in documents:
                doc["rerank_score"] = 0.5  # Neutral score
            return documents[:top_k] if top_k else documents

        # Prepare query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Score in batches
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = self._score_batch(batch)
            all_scores.extend(scores)

        # Attach scores and sort
        for doc, score in zip(documents, all_scores):
            doc["rerank_score"] = score

        # Sort by rerank_score descending
        sorted_docs = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

        result = sorted_docs[:top_k] if top_k else sorted_docs

        logger.info(
            "Re-ranking complete | query='{}' | candidates={} | returned={} | best_score={:.4f}",
            query,
            len(documents),
            len(result),
            result[0]["rerank_score"] if result else 0.0,
        )

        return result

    def _score_batch(self, pairs: List[tuple[str, str]]) -> List[float]:
        """
        Score a batch of (query, document) pairs using the cross-encoder.
        Returns sigmoid-normalized scores in [0, 1].
        """
        assert self._tokenizer is not None and self._model is not None

        # Tokenize with truncation and padding
        inputs = self._tokenizer(
            [p[0] for p in pairs],  # queries
            [p[1] for p in pairs],  # documents
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference (no gradient computation)
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(-1)  # Shape: (batch_size,)

        # Sigmoid to normalize to [0, 1]
        scores = torch.sigmoid(logits).cpu().numpy().tolist()

        return [float(s) for s in scores]

    def _resolve_device(self, device: str) -> str:
        """
        Resolve 'auto' to the best available hardware accelerator.
        Priority: CUDA > MPS (Apple Silicon) > CPU.
        """
        if device != "auto":
            return device

        if torch.cuda.is_available():
            logger.info("CUDA detected for cross-encoder. Using GPU.")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected for cross-encoder. Using Metal.")
            return "mps"
        else:
            logger.info("No GPU detected for cross-encoder. Falling back to CPU.")
            return "cpu"