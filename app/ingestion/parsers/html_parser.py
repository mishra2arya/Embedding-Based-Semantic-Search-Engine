"""HTML document parser using BeautifulSoup."""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.base import BaseParser, ParsedDocument


class HTMLParser(BaseParser):
    """Parses HTML documents, extracting visible text, title, and structure."""

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
        try:
            html_content = raw_doc.content.decode("utf-8")
        except UnicodeDecodeError:
            html_content = raw_doc.content.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = raw_doc.filename
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            h1 = soup.find("h1")
            if h1 and h1.get_text():
                title = h1.get_text().strip()

        # Remove scripts, styles, forms, and navigation
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")
        normalized_text = TextNormalizer.normalize(raw_text)

        metadata = dict(raw_doc.metadata)
        metadata.update(
            {
                "parser": "HTMLParser",
                "format": "html",
            }
        )

        return ParsedDocument(
            title=title,
            text=normalized_text,
            filename=raw_doc.filename,
            mime_type="text/html",
            source=raw_doc.source,
            tenant_id=raw_doc.tenant_id,
            metadata=metadata,
        )
