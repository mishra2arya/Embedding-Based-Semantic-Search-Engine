"""Metadata models, deterministic ID generation, and content deduplication."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Normalized document representation for storage and indexing."""

    document_id: str = Field(description="Deterministic document identifier")
    source: str = Field(default="direct", description="Document origin or collection")
    title: str = Field(default="Untitled", description="Document title")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    language: str = Field(default="en", description="Language code")
    mime_type: str = Field(default="text/plain", description="MIME type")
    checksum: str = Field(description="SHA-256 of normalized full text")
    tenant_id: str = Field(default="default", description="Multi-tenant identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary user metadata")


class ChunkRecord(BaseModel):
    """Individual chunk representation mapped to vector index."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    chunk_index: int = Field(description="Zero-indexed position in document")
    text: str = Field(description="Normalized chunk text")
    token_count: int = Field(default=0, description="Approximate token count")
    character_count: int = Field(default=0, description="Character count")
    checksum: str = Field(description="SHA-256 of chunk text")
    tenant_id: str = Field(
        default="default", description="Tenant identifier inherited from document"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk-level metadata")


class DocumentDeduplicator:
    """Tracks document and chunk checksums for content-based deduplication."""

    def __init__(self):
        self._seen_doc_checksums: set[str] = set()
        self._seen_chunk_checksums: set[str] = set()

    def is_duplicate_doc(self, checksum: str) -> bool:
        """Check if document checksum has already been processed."""
        return checksum in self._seen_doc_checksums

    def register_doc(self, checksum: str) -> None:
        """Mark document checksum as indexed."""
        self._seen_doc_checksums.add(checksum)

    def is_duplicate_chunk(self, chunk_checksum: str) -> bool:
        """Check if chunk checksum has already been processed."""
        return chunk_checksum in self._seen_chunk_checksums

    def register_chunk(self, chunk_checksum: str) -> None:
        """Mark chunk checksum as indexed."""
        self._seen_chunk_checksums.add(chunk_checksum)

    def remove_doc(self, checksum: str) -> None:
        """Remove document checksum on deletion."""
        self._seen_doc_checksums.discard(checksum)

    def clear(self) -> None:
        self._seen_doc_checksums.clear()
        self._seen_chunk_checksums.clear()


def generate_document_id(checksum: str, prefix: str = "doc_") -> str:
    """Generate a deterministic document ID from checksum."""
    return f"{prefix}{checksum[:16]}"


def generate_chunk_id(document_id: str, index: int, prefix: str = "chk_") -> str:
    """Generate a deterministic chunk ID from document ID and chunk index."""
    clean_doc = document_id.replace("doc_", "")
    return f"{prefix}{clean_doc}_{index}"
