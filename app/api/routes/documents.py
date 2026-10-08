"""Document inspection, listing, and deletion endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel

from app.api.dependencies import (
    get_current_user,
    get_index_manager,
    require_operator,
)
from app.core.exceptions import DocumentNotFoundError
from app.core.security import Role
from app.indexing.index_manager import IndexManager

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentItem(BaseModel):
    document_id: str
    title: str
    source: str
    created_at: str
    updated_at: str
    language: str
    mime_type: str
    checksum: str
    tenant_id: str
    metadata: dict[str, Any]


class DocumentListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    documents: list[DocumentItem]


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(
    tenant_id: str | None = Query("default", description="Tenant filter"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
):
    """Retrieve indexed documents with pagination and metadata inspection."""
    if not index_mgr.metadata_store:
        return DocumentListResponse(total=0, offset=offset, limit=limit, documents=[])

    docs = index_mgr.metadata_store.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)
    total = index_mgr.metadata_store.count_documents(tenant_id=tenant_id)

    items = [
        DocumentItem(
            document_id=d["document_id"],
            title=d.get("title", "Untitled"),
            source=d.get("source", "unknown"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            language=d.get("language", "en"),
            mime_type=d.get("mime_type", "text/plain"),
            checksum=d.get("checksum", ""),
            tenant_id=d.get("tenant_id", "default"),
            metadata=d.get("metadata", {}),
        )
        for d in docs
    ]

    return DocumentListResponse(total=total, offset=offset, limit=limit, documents=items)


@router.get("/{document_id}", response_model=DocumentItem, summary="Get document by ID")
def get_document(
    document_id: str,
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
):
    """Fetch complete metadata for a specific document."""
    if not index_mgr.metadata_store:
        raise DocumentNotFoundError(f"Document {document_id} not found.")

    doc = index_mgr.metadata_store.get_document(document_id)
    if not doc:
        raise DocumentNotFoundError(f"Document {document_id} not found.")

    return DocumentItem(
        document_id=doc["document_id"],
        title=doc.get("title", "Untitled"),
        source=doc.get("source", "unknown"),
        created_at=doc.get("created_at", ""),
        updated_at=doc.get("updated_at", ""),
        language=doc.get("language", "en"),
        mime_type=doc.get("mime_type", "text/plain"),
        checksum=doc.get("checksum", ""),
        tenant_id=doc.get("tenant_id", "default"),
        metadata=doc.get("metadata", {}),
    )


@router.delete(
    "/{document_id}", summary="Delete document and tombstone chunks", status_code=status.HTTP_200_OK
)
def delete_document(
    document_id: str,
    tenant_id: str = Query("default", description="Tenant ID"),
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
    _: None = Depends(require_operator),
):
    """Mark document and associated chunks as tombstoned so they are immediately excluded from search."""
    success = index_mgr.delete_document(document_id, tenant_id=tenant_id)
    if not success:
        raise DocumentNotFoundError(f"Document {document_id} not found or tenant mismatch.")

    return {"status": "deleted", "document_id": document_id, "tenant_id": tenant_id}
