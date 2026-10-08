"""PDF document parser using pypdf."""

from __future__ import annotations

import io

from pypdf import PdfReader

from app.core.exceptions import DocumentParsingError
from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.base import BaseParser, ParsedDocument


class PDFParser(BaseParser):
    """Parses PDF documents, extracting text, page count, and embedded metadata."""

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(raw_doc.content))
        except Exception as e:
            raise DocumentParsingError(f"Corrupt or unreadable PDF {raw_doc.filename}: {e}") from e

        pages_text = []
        for _i, page in enumerate(reader.pages):
            try:
                page_content = page.extract_text() or ""
                if page_content.strip():
                    pages_text.append(page_content)
            except Exception:
                continue

        full_text = "\n\n".join(pages_text)
        normalized_text = TextNormalizer.normalize(full_text)

        title = raw_doc.filename
        metadata = dict(raw_doc.metadata)

        if reader.metadata:
            if reader.metadata.title:
                title = reader.metadata.title
            if reader.metadata.author:
                metadata["author"] = reader.metadata.author

        metadata.update(
            {
                "parser": "PDFParser",
                "format": "pdf",
                "page_count": len(reader.pages),
            }
        )

        return ParsedDocument(
            title=title,
            text=normalized_text,
            filename=raw_doc.filename,
            mime_type="application/pdf",
            source=raw_doc.source,
            tenant_id=raw_doc.tenant_id,
            metadata=metadata,
        )
