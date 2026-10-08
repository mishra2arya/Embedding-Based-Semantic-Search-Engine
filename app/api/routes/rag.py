"""RAG API endpoint for contextual question answering and citation attribution."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_rag_pipeline
from app.core.security import Role
from app.observability.activity import activity_logger
from app.observability.metrics import SEARCH_REQUESTS_TOTAL
from app.rag.generator import RAGPipeline, RAGResponse

router = APIRouter(tags=["RAG"])


class RAGRequest(BaseModel):
    query: str = Field(
        description="Question or prompt for RAG generation", min_length=1, max_length=2000
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Top relevant source chunks to consult")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filtering predicates"
    )
    tenant_id: str = Field(default="default", description="Multi-tenant workspace ID")
    rerank: bool = Field(default=True, description="Enable reranking for higher precision context")


@router.post("/rag", response_model=RAGResponse, summary="Generate answer using RAG pipeline")
def execute_rag(
    req: RAGRequest,
    user: tuple[str, Role] = Depends(get_current_user),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Retrieve grounded context, apply prompt-injection defenses, and generate answer with verified citations."""
    try:
        response = pipeline.generate_answer(
            query=req.query,
            top_k=req.top_k,
            tenant_id=req.tenant_id,
            filters=req.filters,
            rerank=req.rerank,
        )
        SEARCH_REQUESTS_TOTAL.labels(
            endpoint="/api/v1/rag", status="success", tenant_id=req.tenant_id
        ).inc()
        activity_logger.log(
            event_type="rag",
            title=f'RAG Synthesis: "{req.query[:45]}"',
            details=f"Generated answer with {len(response.sources)} verified citations",
            latency_ms=response.retrieval_latency_ms + response.generation_latency_ms,
        )
        return response
    except Exception:
        SEARCH_REQUESTS_TOTAL.labels(
            endpoint="/api/v1/rag", status="error", tenant_id=req.tenant_id
        ).inc()
        raise
