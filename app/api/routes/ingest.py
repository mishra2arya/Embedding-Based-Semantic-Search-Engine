"""Ingestion API endpoints for documents and files."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_current_user,
    get_index_manager,
    get_ingestion_pipeline,
    get_search_engine,
    require_operator,
)
from app.core.security import Role, sanitize_filename
from app.indexing.index_manager import IndexManager
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.pipeline import IngestionPipeline
from app.observability.activity import activity_logger
from app.observability.metrics import (
    INDEX_SIZE,
    INGESTION_DOCUMENTS_TOTAL,
    INGESTION_FAILURES_TOTAL,
)
from app.retrieval.hybrid import HybridSearchEngine

router = APIRouter(tags=["Ingestion"])


class IngestTextRequest(BaseModel):
    text: str = Field(description="Raw document content", min_length=1)
    title: str = Field(default="Untitled Document", description="Document title")
    source: str = Field(default="api_direct", description="Document origin")
    tenant_id: str = Field(default="default", description="Tenant ID")
    metadata: dict[str, Any] | None = Field(default=None, description="Custom metadata attributes")


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_created: int
    is_duplicate: bool
    tenant_id: str


@router.post("/ingest", response_model=IngestResponse, summary="Ingest raw text or document")
def ingest_text(
    req: IngestTextRequest,
    user: tuple[str, Role] = Depends(get_current_user),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    index_mgr: IndexManager = Depends(get_index_manager),
    engine: HybridSearchEngine = Depends(get_search_engine),
    _: None = Depends(require_operator),
):
    """Parse, chunk, embed, and index a text document with automatic deduplication."""
    try:
        res = pipeline.process_text(
            text=req.text,
            title=req.title,
            source=req.source,
            tenant_id=req.tenant_id,
            metadata=req.metadata,
        )

        if res.is_duplicate:
            return IngestResponse(
                status="duplicate_skipped",
                document_id=res.document.document_id,
                chunks_created=0,
                is_duplicate=True,
                tenant_id=req.tenant_id,
            )

        if res.chunks:
            # Generate embeddings
            texts = [c.text for c in res.chunks]
            vectors = engine.embedding_service.encode(texts, normalize=True, use_cache=False)

            # Insert into FAISS & metadata store
            index_mgr.add_batch([res.document], res.chunks, vectors)

            # Sync BM25
            engine.sync_lexical_index(tenant_id=req.tenant_id)

            if index_mgr.faiss_wrapper:
                INDEX_SIZE.set(index_mgr.faiss_wrapper.total_vectors)

        INGESTION_DOCUMENTS_TOTAL.labels(tenant_id=req.tenant_id, status="success").inc()
        activity_logger.log(
            event_type="ingest",
            title=f'Ingested Document: "{req.title[:45]}"',
            details=f"Created {len(res.chunks)} chunks (tenant: {req.tenant_id})",
        )

        return IngestResponse(
            status="indexed",
            document_id=res.document.document_id,
            chunks_created=len(res.chunks),
            is_duplicate=False,
            tenant_id=req.tenant_id,
        )

    except Exception as e:
        INGESTION_FAILURES_TOTAL.labels(reason=type(e).__name__).inc()
        raise


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    summary="Upload and ingest file (PDF, DOCX, TXT, HTML, JSON, CSV)",
)
async def ingest_file(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    source: str = Form("file_upload"),
    tenant_id: str = Form("default"),
    user: tuple[str, Role] = Depends(get_current_user),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    index_mgr: IndexManager = Depends(get_index_manager),
    engine: HybridSearchEngine = Depends(get_search_engine),
    _: None = Depends(require_operator),
):
    """Upload a document file (PDF, Word, Markdown, HTML, JSON, CSV) for automated ingestion."""
    content = await file.read()
    safe_filename = sanitize_filename(file.filename or "upload.txt")
    doc_title = title or safe_filename

    raw_doc = RawDocument(
        content=content,
        filename=safe_filename,
        mime_type=file.content_type,
        source=source,
        tenant_id=tenant_id,
        metadata={"original_filename": file.filename},
    )

    res = pipeline.process_raw_document(raw_doc)
    res.document.title = doc_title

    if res.is_duplicate:
        return IngestResponse(
            status="duplicate_skipped",
            document_id=res.document.document_id,
            chunks_created=0,
            is_duplicate=True,
            tenant_id=tenant_id,
        )

    if res.chunks:
        texts = [c.text for c in res.chunks]
        vectors = engine.embedding_service.encode(texts, normalize=True, use_cache=False)
        index_mgr.add_batch([res.document], res.chunks, vectors)
        engine.sync_lexical_index(tenant_id=tenant_id)
        if index_mgr.faiss_wrapper:
            INDEX_SIZE.set(index_mgr.faiss_wrapper.total_vectors)

    INGESTION_DOCUMENTS_TOTAL.labels(tenant_id=tenant_id, status="success").inc()
    activity_logger.log(
        event_type="ingest",
        title=f'Uploaded File: "{safe_filename}"',
        details=f"Extracted {len(res.chunks)} chunks (tenant: {tenant_id})",
    )

    return IngestResponse(
        status="indexed",
        document_id=res.document.document_id,
        chunks_created=len(res.chunks),
        is_duplicate=False,
        tenant_id=tenant_id,
    )
