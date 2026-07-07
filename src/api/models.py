"""
src/api/models.py
Pydantic request/response schemas for the FastAPI application.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    documents: List[Dict] = Field(
        ...,
        min_length=1,
        description="List of documents to ingest. Each must have 'text' and optionally 'metadata'.",
    )


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)
    filter: Optional[Dict] = Field(default=None)


class SourceReference(BaseModel):
    source: str
    chunk_index: int
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    latency_ms: float
    token_usage: int
    retrieved_chunks: int
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    mode: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_count: int
    embedding_model: str
    llm_model: str