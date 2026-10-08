"""Unit tests for all 4 dynamic chunking strategies."""

from app.ingestion.chunking.factory import ChunkerFactory
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.chunking.sentence_chunker import SentenceChunker
from app.ingestion.chunking.structure_chunker import StructureChunker
from app.ingestion.chunking.token_chunker import TokenChunker


def test_token_chunker():
    chunker = TokenChunker(target_tokens=50, overlap_tokens=10)
    text = "word " * 200
    chunks = chunker.split(text, document_id="doc_1")
    assert len(chunks) > 1
    for c in chunks:
        assert c.document_id == "doc_1"
        assert c.token_count > 0
        assert c.chunk_id.startswith("chk_")


def test_sentence_chunker():
    chunker = SentenceChunker(target_tokens=30, min_tokens=10, overlap_tokens=5)
    text = (
        "Zero Trust is a security paradigm. It verifies every request explicitly. "
        "Network microsegmentation isolates critical assets. Transport encryption protects data. "
        "Continuous monitoring detects anomalies."
    )
    chunks = chunker.split(text, document_id="doc_2")
    assert len(chunks) >= 1
    assert all("." in c.text for c in chunks)


def test_semantic_chunker():
    chunker = SemanticChunker(target_tokens=40, min_tokens=10, similarity_threshold=0.6)
    text = (
        "Cryptography secures network channels using mathematical ciphers. "
        "Diffie Hellman performs secret key negotiation over untrusted media. "
        "Baking sourdough bread requires flour water and wild yeast fermentation. "
        "Bread crust develops through Maillard browning reactions."
    )
    chunks = chunker.split(text, document_id="doc_3")
    assert len(chunks) >= 1
    assert all(c.token_count > 0 for c in chunks)


def test_structure_chunker():
    chunker = StructureChunker(target_tokens=40, min_tokens=10)
    text = (
        "# Security Architecture\n\n"
        "Mutual TLS enforces bidirectional certificates between microservices.\n\n"
        "```python\ndef verify():\n    return True\n```\n\n"
        "## Authorization Rules\n\n"
        "Role-Based Access Control restricts operations by user role."
    )
    chunks = chunker.split(text, document_id="doc_4")
    assert len(chunks) >= 1
    # Check that code block was preserved intact
    combined = " ".join(c.text for c in chunks)
    assert "def verify():" in combined


def test_chunker_factory():
    for name, expected_cls in [
        ("token", TokenChunker),
        ("sentence", SentenceChunker),
        ("semantic", SemanticChunker),
        ("structure", StructureChunker),
    ]:
        instance = ChunkerFactory.get_chunker(strategy=name)
        assert isinstance(instance, expected_cls)
