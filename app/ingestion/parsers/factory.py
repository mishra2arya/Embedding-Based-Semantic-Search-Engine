"""Parser factory resolving documents to parsers."""

from __future__ import annotations

import os

from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.csv_parser import CSVParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.html_parser import HTMLParser
from app.ingestion.parsers.json_parser import JSONParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.text_parser import TextParser


class ParserFactory:
    """Factory to get the right parser for a given file extension or MIME type."""

    _EXTENSION_MAP: dict[str, type[BaseParser]] = {
        ".txt": TextParser,
        ".md": TextParser,
        ".markdown": TextParser,
        ".html": HTMLParser,
        ".htm": HTMLParser,
        ".json": JSONParser,
        ".jsonl": JSONParser,
        ".csv": CSVParser,
        ".pdf": PDFParser,
        ".docx": DOCXParser,
    }

    _MIME_MAP: dict[str, type[BaseParser]] = {
        "text/plain": TextParser,
        "text/markdown": TextParser,
        "text/html": HTMLParser,
        "application/json": JSONParser,
        "text/csv": CSVParser,
        "application/pdf": PDFParser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCXParser,
    }

    @classmethod
    def get_parser(cls, filename: str, mime_type: str | None = None) -> BaseParser:
        """Resolve appropriate parser instance."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in cls._EXTENSION_MAP:
            return cls._EXTENSION_MAP[ext]()

        if mime_type and mime_type.lower() in cls._MIME_MAP:
            return cls._MIME_MAP[mime_type.lower()]()

        # Fallback to plain text parser
        return TextParser()
