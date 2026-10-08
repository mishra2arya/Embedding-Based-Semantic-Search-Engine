"""Base parser interface and parsed document model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.loaders.file_loader import RawDocument


@dataclass
class ParsedDocument:
    """Document representation after parsing and normalization."""

    title: str
    text: str
    filename: str
    mime_type: str
    source: str
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        """Extract text and metadata from raw document."""
        pass
