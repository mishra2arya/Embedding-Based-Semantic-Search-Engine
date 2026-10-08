"""Base chunker interface and chunk model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.metadata import ChunkRecord, generate_chunk_id


class BaseChunker(ABC):
    """Abstract interface for all chunking strategies."""

    def __init__(
        self,
        target_tokens: int = 350,
        min_tokens: int = 100,
        max_tokens: int = 600,
        overlap_tokens: int = 50,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def split(self, text: str, document_id: str, tenant_id: str = "default") -> list[ChunkRecord]:
        """Split document text into a list of ChunkRecord instances."""
        pass

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (approximately 1 token ≈ 4 characters or 0.75 words)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def create_chunk(
        self,
        text: str,
        document_id: str,
        chunk_index: int,
        tenant_id: str = "default",
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkRecord:
        """Factory method to construct a ChunkRecord."""
        normalized = TextNormalizer.normalize(text)
        checksum = TextNormalizer.compute_sha256(normalized)
        meta = extra_metadata or {}
        meta["strategy"] = self.__class__.__name__

        return ChunkRecord(
            chunk_id=generate_chunk_id(document_id, chunk_index),
            document_id=document_id,
            chunk_index=chunk_index,
            text=normalized,
            token_count=self.estimate_tokens(normalized),
            character_count=len(normalized),
            checksum=checksum,
            tenant_id=tenant_id,
            metadata=meta,
        )
