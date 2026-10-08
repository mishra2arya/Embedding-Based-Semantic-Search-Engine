"""Text and Markdown document parser."""

from __future__ import annotations

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.base import BaseParser, ParsedDocument


class TextParser(BaseParser):
    """Parses plain text (.txt) and Markdown (.md) documents."""

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        try:
            text = raw_doc.content.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_doc.content.decode("latin-1", errors="replace")

        # Extract title: prefer explicit metadata, then markdown header, then filename
        title = raw_doc.metadata.get("title") or raw_doc.filename
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            first_line = lines[0]
            if first_line.startswith("# "):
                title = first_line.lstrip("# ").strip()
            elif not raw_doc.metadata.get("title") and len(first_line) < 60:
                title = first_line

        # Remove markdown frontmatter if present
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2]

        normalized_text = TextNormalizer.normalize(text)

        metadata = dict(raw_doc.metadata)
        metadata.update(
            {
                "parser": "TextParser",
                "format": "markdown" if raw_doc.filename.endswith(".md") else "text",
            }
        )

        return ParsedDocument(
            title=title,
            text=normalized_text,
            filename=raw_doc.filename,
            mime_type=raw_doc.mime_type or "text/plain",
            source=raw_doc.source,
            tenant_id=raw_doc.tenant_id,
            metadata=metadata,
        )
