"""CSV tabular document parser."""

from __future__ import annotations

import csv
import io

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.base import BaseParser, ParsedDocument


class CSVParser(BaseParser):
    """Parses CSV documents into structured text blocks."""

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        content_str = raw_doc.content.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(content_str))

        rows = list(reader)
        if not rows:
            return ParsedDocument(
                title=raw_doc.filename,
                text="",
                filename=raw_doc.filename,
                mime_type="text/csv",
                source=raw_doc.source,
                tenant_id=raw_doc.tenant_id,
            )

        header = rows[0]
        row_texts = []
        for row in rows[1:]:
            if not any(row):
                continue
            row_items = [
                f"{header[i]}: {row[i]}"
                for i in range(min(len(header), len(row)))
                if row[i].strip()
            ]
            row_texts.append(" | ".join(row_items))

        text = "\n".join(row_texts)
        normalized_text = TextNormalizer.normalize(text)

        metadata = dict(raw_doc.metadata)
        metadata.update(
            {
                "parser": "CSVParser",
                "format": "csv",
                "row_count": len(rows) - 1,
                "columns": header,
            }
        )

        return ParsedDocument(
            title=raw_doc.filename,
            text=normalized_text,
            filename=raw_doc.filename,
            mime_type="text/csv",
            source=raw_doc.source,
            tenant_id=raw_doc.tenant_id,
            metadata=metadata,
        )
