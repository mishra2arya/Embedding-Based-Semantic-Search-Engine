"""Unit tests for multi-format document parsers."""

import json

from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.parsers.csv_parser import CSVParser
from app.ingestion.parsers.factory import ParserFactory
from app.ingestion.parsers.html_parser import HTMLParser
from app.ingestion.parsers.json_parser import JSONParser
from app.ingestion.parsers.text_parser import TextParser


def test_text_parser():
    raw = RawDocument(
        content=b"# Header Title\n\nBody paragraph text.",
        filename="sample.md",
        mime_type="text/markdown",
    )
    parser = ParserFactory.get_parser("sample.md")
    assert isinstance(parser, TextParser)
    parsed = parser.parse(raw)
    assert parsed.title == "Header Title"
    assert "Body paragraph text." in parsed.text


def test_html_parser():
    html = b"<html><head><title>HTML Page Title</title><script>var x = 1;</script></head><body><h1>Main Heading</h1><p>Visible content paragraph.</p></body></html>"
    raw = RawDocument(content=html, filename="page.html", mime_type="text/html")
    parser = ParserFactory.get_parser("page.html")
    assert isinstance(parser, HTMLParser)
    parsed = parser.parse(raw)
    assert parsed.title == "HTML Page Title"
    assert "Visible content paragraph." in parsed.text
    assert "var x = 1" not in parsed.text


def test_json_parser():
    doc = {"title": "JSON Record", "text": "Extracted JSON content field.", "category": "cloud"}
    raw = RawDocument(
        content=json.dumps(doc).encode("utf-8"), filename="data.json", mime_type="application/json"
    )
    parser = ParserFactory.get_parser("data.json")
    assert isinstance(parser, JSONParser)
    parsed = parser.parse(raw)
    assert parsed.title == "JSON Record"
    assert "Extracted JSON content field." in parsed.text
    assert parsed.metadata.get("category") == "cloud"


def test_csv_parser():
    csv_bytes = b"service,protocol,port\nhttps,tcp,443\nssh,tcp,22"
    raw = RawDocument(content=csv_bytes, filename="ports.csv", mime_type="text/csv")
    parser = ParserFactory.get_parser("ports.csv")
    assert isinstance(parser, CSVParser)
    parsed = parser.parse(raw)
    assert "https" in parsed.text
    assert "443" in parsed.text
