"""
src/api/routes.py
FastAPI route handlers. Dependencies come from app.state (set in lifespan).
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from loguru import logger

from src.api.models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceReference,
)
from src.ingestion.loader import Document
from src.ingestion.chunker import RecursiveCharacterTextSplitter


router = APIRouter()


# ------------------------------------------------------------------
# Helper: get services from app.state
# ------------------------------------------------------------------
def _get_services(request: Request):
    """Extract initialized services from app.state."""
    return {
        "embedder": request.app.state.embedder,
        "store": request.app.state.vector_store,
        "engine": request.app.state.hybrid_engine,
        "reranker": request.app.state.reranker,
        "rag_chain": request.app.state.rag_chain,
    }


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """System health check."""
    services = _get_services(request)
    store = services["store"]
    embedder = services["embedder"]
    rag_chain = services["rag_chain"]

    from config.settings import get_settings

    settings = get_settings()

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vector_store_count=store.count(),
        embedding_model=embedder.model_name,
        llm_model=rag_chain.model_name,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest documents into the vector store."""
    logger.info("Ingestion request | documents={}", len(body.documents))

    services = _get_services(request)
    embedder = services["embedder"]
    store = services["store"]

    # Convert to internal Document format and chunk
    raw_docs = [
        Document(text=d.get("text", ""), source=d.get("source", "api"), metadata=d.get("metadata", {}))
        for d in body.documents
    ]

    chunker = RecursiveCharacterTextSplitter()
    chunks = chunker.split_documents(raw_docs)

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text found in documents")

    # Embed and store
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    metadatas = [c.metadata for c in chunks]
    store.add_documents(texts, embeddings.tolist(), metadatas)

    return IngestResponse(
        status="success",
        documents_ingested=len(body.documents),
        chunks_created=len(chunks),
        message=f"Ingested {len(body.documents)} documents into {len(chunks)} chunks",
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Full RAG query: retrieve, re-rank, and generate an answer."""
    logger.info("Query request | question='{}' | top_k={}", body.question, body.top_k)

    services = _get_services(request)
    embedder = services["embedder"]
    engine = services["engine"]
    reranker = services["reranker"]
    rag_chain = services["rag_chain"]

    # Embed query
    raw_embedding = embedder.embed_single(body.question)
    query_embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)

    # Retrieve
    results = engine.search_hybrid(
        query=body.question,
        query_embedding=query_embedding,
        where=body.filter,
    )

    # Re-rank if enabled
    if body.rerank and results:
        results = reranker.rerank(body.question, results, top_k=body.top_k)

    # Generate
    response = rag_chain.generate(body.question, results)

    # Format sources
    sources = [
        SourceReference(
            source=s["source"],
            chunk_index=s["chunk_index"],
            text_preview=s["text_preview"],
        )
        for s in response["sources"]
    ]

    return QueryResponse(
        answer=response["answer"],
        sources=sources,
        latency_ms=response["latency_ms"],
        token_usage=response["token_usage"],
        retrieved_chunks=len(results),
        error=response["error"],
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Retrieval-only search. No LLM generation."""
    logger.info("Search request | query='{}' | mode={}", body.query, body.mode)

    services = _get_services(request)
    embedder = services["embedder"]
    engine = services["engine"]

    raw_embedding = embedder.embed_single(body.query)
    query_embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)

    if body.mode == "dense":
        results = engine.search_dense(query_embedding, top_k=body.top_k)
    elif body.mode == "sparse":
        results = engine.search_sparse(body.query, top_k=body.top_k)
    else:
        results = engine.search_hybrid(
            query=body.query,
            query_embedding=query_embedding,
        )

    formatted = [
        SearchResult(
            id=r["id"],
            text=r["text"][:500],
            metadata=r.get("metadata", {}),
            score=r.get("rerank_score", r.get("rrf_score", 0.0)),
        )
        for r in results
    ]

    return SearchResponse(
        results=formatted,
        query=body.query,
        mode=body.mode,
    )