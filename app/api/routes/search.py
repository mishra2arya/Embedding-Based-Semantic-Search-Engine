"""Search API endpoints supporting semantic, lexical, and hybrid retrieval."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_search_engine
from app.core.security import Role
from app.observability.activity import activity_logger
from app.observability.metrics import (
    EMBEDDING_LATENCY_SECONDS,
    FAISS_LATENCY_SECONDS,
    RERANKER_LATENCY_SECONDS,
    SEARCH_LATENCY_SECONDS,
    SEARCH_REQUESTS_TOTAL,
)
from app.retrieval.hybrid import HybridSearchEngine

router = APIRouter(tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(description="Natural language query string", min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filtering predicates"
    )
    tenant_id: str = Field(default="default", description="Multi-tenant workspace ID")
    mode: str = Field(default="hybrid", description="Retrieval mode: hybrid, semantic, lexical")
    rerank: bool = Field(default=False, description="Enable optional second-stage reranking")
    fusion_method: str = Field(default="weighted", description="Fusion strategy: weighted or rrf")


class SearchResultItem(BaseModel):
    rank: int
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    final_score: float
    document_id: str
    chunk_id: str
    title: str
    text: str
    metadata: dict[str, Any]
    citation: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    latency_ms: float
    mode: str | None = None
    reranked: bool | None = None
    total_candidates: int | None = None
    latency_profile: dict[str, float] | None = None


@router.post("/search", response_model=SearchResponse, summary="Execute semantic / hybrid search")
def execute_search(
    req: SearchRequest,
    user: tuple[str, Role] = Depends(get_current_user),
    engine: HybridSearchEngine = Depends(get_search_engine),
):
    """Execute low-latency dense vector, lexical, or hybrid search with explainable scores."""
    t0 = time.perf_counter()

    try:
        search_out = engine.search(
            query=req.query,
            top_k=req.top_k,
            filters=req.filters,
            tenant_id=req.tenant_id,
            mode=req.mode,
            rerank=req.rerank,
            fusion_method=req.fusion_method,
        )

        # Record Prometheus metrics
        elapsed_sec = time.perf_counter() - t0
        SEARCH_LATENCY_SECONDS.observe(elapsed_sec)

        profile = search_out.get("latency_profile", {})
        if "embedding_generation_ms" in profile:
            EMBEDDING_LATENCY_SECONDS.observe(profile["embedding_generation_ms"] / 1000.0)
        if "faiss_retrieval_ms" in profile:
            FAISS_LATENCY_SECONDS.observe(profile["faiss_retrieval_ms"] / 1000.0)
        if "reranking_ms" in profile and req.rerank:
            RERANKER_LATENCY_SECONDS.observe(profile["reranking_ms"] / 1000.0)

        SEARCH_REQUESTS_TOTAL.labels(
            endpoint="/api/v1/search", status="success", tenant_id=req.tenant_id
        ).inc()

        activity_logger.log(
            event_type="search",
            title=f'Query: "{req.query[:45]}"',
            details=f"Retrieved {len(search_out['results'])} results ({req.mode}, top_k={req.top_k})",
            latency_ms=search_out["latency_ms"],
        )

        return SearchResponse(
            query=search_out["query"],
            results=search_out["results"],
            latency_ms=search_out["latency_ms"],
            mode=search_out["mode"],
            reranked=search_out["reranked"],
            total_candidates=search_out["total_candidates"],
            latency_profile=search_out["latency_profile"],
        )

    except Exception:
        SEARCH_REQUESTS_TOTAL.labels(
            endpoint="/api/v1/search", status="error", tenant_id=req.tenant_id
        ).inc()
        raise
