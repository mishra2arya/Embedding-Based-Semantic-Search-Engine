"""Document loader interfaces and filesystem loaders."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.exceptions import PayloadTooLargeError
from app.core.security import sanitize_filename


@dataclass
class RawDocument:
    """Represents raw loaded document data before parsing."""

    content: bytes
    filename: str
    mime_type: str | None = None
    source: str = "filesystem"
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: Path | str) -> Iterator[RawDocument]:
        """Yield raw documents from the specified source."""
        pass


class FileSystemLoader(BaseLoader):
    """Loads files from directory or single file with path traversal defense and size limits."""

    def __init__(self, max_size_mb: int = 25, allowed_extensions: list[str] | None = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (
                allowed_extensions
                or [".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".csv", ".pdf", ".docx"]
            )
        }

    def load(self, source: Path | str) -> Iterator[RawDocument]:
        path = Path(source).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            yield self._load_file(path)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in self.allowed_extensions:
                        try:
                            yield self._load_file(file_path)
                        except Exception:
                            # Gracefully continue if an individual file is corrupt
                            continue

    def _load_file(self, file_path: Path) -> RawDocument:
        safe_name = sanitize_filename(file_path.name)
        size = file_path.stat().st_size
        if size > self.max_size_bytes:
            raise PayloadTooLargeError(
                f"File {file_path.name} ({size / (1024 * 1024):.2f}MB) exceeds limit of {self.max_size_bytes / (1024 * 1024):.2f}MB"
            )

        with open(file_path, "rb") as f:
            content = f.read()

        return RawDocument(
            content=content,
            filename=safe_name,
            source=str(file_path),
            tenant_id="default",
            metadata={"file_size": size, "path": str(file_path)},
        )
