"""
RAG Generation Chain: assembles context, builds the prompt, and calls the LLM.

Design decisions:
- The system prompt is strict: "Answer ONLY from the provided context."
  This minimizes hallucination and grounds the response in retrieved evidence.
- Context is formatted with clear delimiters and source annotations so the
  LLM can cite sources accurately.
- We truncate context from the bottom if it exceeds the token budget,
  preserving the highest-re-ranked chunks at the top.
- OpenAI is the default provider, but the architecture supports swapping
  to Ollama or any other HTTP-compatible LLM.
- If the LLM call fails, we return a graceful error with the raw context
  so the user can still see what was retrieved.
- All API calls use httpx for async-ready, timeout-aware HTTP.
"""

import time
from typing import Dict, List, Optional

import httpx
from loguru import logger

from config.settings import get_settings


class RAGGenerationError(Exception):
    """Custom exception for RAG generation failures."""
    pass


class RAGChain:
    """
    End-to-end RAG generation chain.
    Takes retrieved documents, builds a structured prompt, and calls the LLM.
    """

    SYSTEM_PROMPT = """You are a precise, enterprise-grade research assistant.
Your task is to answer the user's question using ONLY the provided context.
You must NOT use outside knowledge, speculation, or inference beyond what is explicitly stated in the context.

Rules:
1. Answer concisely and accurately.
2. Cite your sources inline using the format [Source: filename, chunk_index].
3. If the context does not contain the answer, say "I cannot answer this based on the provided documents."
4. Do not mention that you are an AI or that you have limited context.
5. At the end of your answer, list all sources you used in a "References" section."""

    CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with citations):"""

    MAX_CONTEXT_TOKENS_ESTIMATE = 3000  # Conservative budget for GPT-4o-mini

    def __init__(self):
        settings = get_settings()
        self.provider = settings.llm_provider
        self.model_name = settings.llm_model_name
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        self._http_client = httpx.Client(
            timeout=60.0,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )

        logger.info(
            "RAGChain initialized | provider={} | model={} | temp={}",
            self.provider,
            self.model_name,
            self.temperature,
        )

    def generate(
        self,
        question: str,
        documents: List[Dict],
    ) -> Dict:
        """
        Generate an answer from retrieved documents.

        Args:
            question: The user's natural language question.
            documents: List of retrieved/re-ranked document dicts. Each must have
                       'text', 'metadata', and optionally 'rerank_score'.

        Returns:
            Dict with keys: answer, sources, latency_ms, token_usage, error.
        """
        if not documents:
            return {
                "answer": "I cannot answer this based on the provided documents.",
                "sources": [],
                "latency_ms": 0.0,
                "token_usage": 0,
                "error": None,
            }

        start_time = time.perf_counter()

        # 1. Format context with source annotations
        context_blocks = self._format_context(documents)

        # 2. Truncate if too long (preserve highest-ranked first)
        context_text = self._truncate_context(context_blocks)

        # 3. Build the full prompt
        prompt = self.CONTEXT_TEMPLATE.format(
            context=context_text,
            question=question,
        )

        # 4. Call LLM
        try:
            answer, token_usage = self._call_llm(prompt)
            error = None
        except Exception as e:
            logger.exception("LLM generation failed | error={}", e)
            answer = (
                "I encountered an error while generating the answer. "
                "Here is the relevant context I found:\n\n" + context_text
            )
            token_usage = 0
            error = str(e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 5. Extract unique sources from documents
        sources = self._extract_sources(documents)

        result = {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
            "token_usage": token_usage,
            "error": error,
        }

        logger.info(
            "Generation complete | latency_ms={} | tokens={} | error={}",
            result["latency_ms"],
            token_usage,
            error is not None,
        )

        return result

    def _format_context(self, documents: List[Dict]) -> List[str]:
        """
        Format each document as a numbered context block with source metadata.
        """
        blocks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.get("metadata", {}).get("source", "unknown")
            chunk_idx = doc.get("metadata", {}).get("chunk_index", 0)
            text = doc.get("text", "").strip()

            block = (
                f"[{i}] Source: {source} (chunk {chunk_idx})\n"
                f"{text}"
            )
            blocks.append(block)

        return blocks

    def _truncate_context(self, context_blocks: List[str]) -> str:
        """
        Join context blocks and truncate if they exceed token budget.
        We use a rough heuristic: 1 token ≈ 4 characters for English.
        """
        full_text = "\n\n".join(context_blocks)
        estimated_tokens = len(full_text) // 4

        if estimated_tokens <= self.MAX_CONTEXT_TOKENS_ESTIMATE:
            return full_text

        # Truncate from the bottom, keeping highest-ranked chunks
        logger.warning(
            "Context too long | estimated_tokens={} | max={}. Truncating.",
            estimated_tokens,
            self.MAX_CONTEXT_TOKENS_ESTIMATE,
        )

        truncated_blocks = []
        current_chars = 0
        max_chars = self.MAX_CONTEXT_TOKENS_ESTIMATE * 4

        for block in context_blocks:
            if current_chars + len(block) > max_chars:
                break
            truncated_blocks.append(block)
            current_chars += len(block) + 2  # +2 for "\n\n"

        return "\n\n".join(truncated_blocks)

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        """
        Call the LLM API and return (answer_text, token_usage).
        """
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            raise RAGGenerationError(f"Unsupported LLM provider: {self.provider}")

    def _call_openai(self, prompt: str) -> tuple[str, int]:
        """
        Call OpenAI Chat Completions API via httpx.
        """
        if not self.api_key:
            raise RAGGenerationError("OPENAI_API_KEY is not set")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self._http_client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        token_usage = data.get("usage", {}).get("total_tokens", 0)

        return answer, token_usage

    def _call_ollama(self, prompt: str) -> tuple[str, int]:
        """
        Call local Ollama API. Token usage is not tracked by Ollama, so we return 0.
        """
        settings = get_settings()
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "system": self.SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = self._http_client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()
        return answer, 0

    def _extract_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract unique source references from document metadata.
        """
        seen = set()
        sources = []
        for doc in documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            chunk_idx = doc.get("metadata", {}).get("chunk_index", 0)
            key = (source, chunk_idx)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "chunk_index": chunk_idx,
                    "text_preview": doc.get("text", "")[:200],
                })
        return sources