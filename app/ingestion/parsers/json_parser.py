"""JSON and JSON Lines parser."""

from __future__ import annotations

import json

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.base import BaseParser, ParsedDocument


class JSONParser(BaseParser):
    """Parses JSON and JSONL records."""

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        content_str = raw_doc.content.decode("utf-8", errors="replace")

        title = raw_doc.filename
        text_parts = []
        extracted_metadata = dict(raw_doc.metadata)

        # Handle JSON Lines
        if (
            raw_doc.filename.endswith(".jsonl")
            or "\n" in content_str.strip()
            and not content_str.strip().startswith("[")
        ):
            for line in content_str.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        text_val = (
                            record.get("text")
                            or record.get("content")
                            or record.get("body")
                            or json.dumps(record)
                        )
                        text_parts.append(str(text_val))
                except Exception:
                    continue
            full_text = "\n\n".join(text_parts)
        else:
            try:
                data = json.loads(content_str)
                if isinstance(data, dict):
                    title = data.get("title", raw_doc.filename)
                    # Use common text field names
                    for key in ["text", "content", "body", "description", "summary"]:
                        if key in data and isinstance(data[key], str):
                            text_parts.append(data[key])
                    # If no standard text field found, stringify all values
                    if not text_parts:
                        text_parts = [
                            f"{k}: {v}" for k, v in data.items() if not isinstance(v, (dict, list))
                        ]
                    # Extract document-level metadata fields if present
                    if "metadata" in data and isinstance(data["metadata"], dict):
                        extracted_metadata.update(data["metadata"])
                    for field in ["author", "category", "language", "tags", "created_at"]:
                        if field in data:
                            extracted_metadata[field] = data[field]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text_parts.append(
                                item.get("text") or item.get("content") or json.dumps(item)
                            )
                        else:
                            text_parts.append(str(item))
                full_text = "\n\n".join(text_parts) if text_parts else content_str
            except json.JSONDecodeError:
                full_text = content_str

        normalized_text = TextNormalizer.normalize(full_text)
        extracted_metadata.update(
            {
                "parser": "JSONParser",
                "format": "json",
            }
        )

        return ParsedDocument(
            title=title,
            text=normalized_text,
            filename=raw_doc.filename,
            mime_type="application/json",
            source=raw_doc.source,
            tenant_id=raw_doc.tenant_id,
            metadata=extracted_metadata,
        )
