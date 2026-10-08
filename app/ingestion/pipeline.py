"""Ingestion pipeline orchestrating validation, parsing, normalization, deduplication, and chunking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import DocumentParsingError, PayloadTooLargeError
from app.ingestion.chunking.base import BaseChunker
from app.ingestion.chunking.factory import ChunkerFactory
from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import FileSystemLoader, RawDocument
from app.ingestion.metadata import (
    ChunkRecord,
    DocumentDeduplicator,
    DocumentRecord,
    generate_document_id,
)
from app.ingestion.parsers.factory import ParserFactory


@dataclass
class IngestionResult:
    """Result of processing a single document through the ingestion pipeline."""

    document: DocumentRecord
    chunks: list[ChunkRecord]
    is_duplicate: bool = False


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        deduplicator: DocumentDeduplicator | None = None,
        max_file_size_mb: int = 25,
    ):
        self.chunker = chunker or ChunkerFactory.get_chunker(
            strategy=settings.chunking_strategy,
            target_tokens=settings.chunking_target_tokens,
            min_tokens=settings.chunking_min_tokens,
            max_tokens=settings.chunking_max_tokens,
            overlap_tokens=settings.chunking_overlap_tokens,
        )
        self.deduplicator = deduplicator or DocumentDeduplicator()
        self.max_file_size_mb = max_file_size_mb

    def process_raw_document(self, raw_doc: RawDocument) -> IngestionResult:
        """Process a loaded raw document into normalized document record and chunks."""
        # 1. Validation
        if len(raw_doc.content) > self.max_file_size_mb * 1024 * 1024:
            raise PayloadTooLargeError(
                f"Document {raw_doc.filename} exceeds max size {self.max_file_size_mb}MB"
            )

        # 2. Parser
        parser = ParserFactory.get_parser(raw_doc.filename, raw_doc.mime_type)
        try:
            parsed = parser.parse(raw_doc)
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse {raw_doc.filename}: {e}") from e

        # 3. Normalization
        normalized_text = TextNormalizer.normalize(parsed.text)
        if not normalized_text:
            raise DocumentParsingError(
                f"Document {raw_doc.filename} contained no readable text after parsing."
            )

        # 4. Checksum & Deduplication
        checksum = TextNormalizer.compute_sha256(normalized_text)
        doc_id = generate_document_id(checksum)

        if self.deduplicator.is_duplicate_doc(checksum):
            # Duplicate document detected
            doc_record = DocumentRecord(
                document_id=doc_id,
                source=parsed.source,
                title=parsed.title,
                language=raw_doc.metadata.get("language", "en"),
                mime_type=parsed.mime_type,
                checksum=checksum,
                tenant_id=parsed.tenant_id,
                metadata=parsed.metadata,
            )
            return IngestionResult(document=doc_record, chunks=[], is_duplicate=True)

        # Register document checksum
        self.deduplicator.register_doc(checksum)

        # 5. Metadata extraction
        doc_metadata = dict(parsed.metadata)
        doc_record = DocumentRecord(
            document_id=doc_id,
            source=parsed.source,
            title=parsed.title,
            language=doc_metadata.get("language", "en"),
            mime_type=parsed.mime_type,
            checksum=checksum,
            tenant_id=parsed.tenant_id,
            metadata=doc_metadata,
        )

        # 6. Dynamic Chunking
        chunks = self.chunker.split(
            text=normalized_text,
            document_id=doc_id,
            tenant_id=parsed.tenant_id,
        )

        # Deduplicate chunks if identical chunks exist within document
        unique_chunks: list[ChunkRecord] = []
        seen_chunk_hashes: set[str] = set()
        for chk in chunks:
            if chk.checksum not in seen_chunk_hashes:
                seen_chunk_hashes.add(chk.checksum)
                unique_chunks.append(chk)

        return IngestionResult(document=doc_record, chunks=unique_chunks, is_duplicate=False)

    def process_text(
        self,
        text: str,
        title: str = "Untitled",
        source: str = "direct_api",
        tenant_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Process raw text directly from API or memory."""
        meta = dict(metadata or {})
        meta["title"] = title
        raw = RawDocument(
            content=text.encode("utf-8"),
            filename=f"{title.replace(' ', '_').lower()}.txt",
            mime_type="text/plain",
            source=source,
            tenant_id=tenant_id,
            metadata=meta,
        )
        return self.process_raw_document(raw)

    def process_directory(self, dir_path: Path | str) -> Iterator[IngestionResult]:
        """Stream documents from directory through pipeline."""
        loader = FileSystemLoader(max_size_mb=self.max_file_size_mb)
        for raw_doc in loader.load(dir_path):
            try:
                yield self.process_raw_document(raw_doc)
            except Exception:
                continue
