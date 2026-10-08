"""Unit tests for text normalization, hashing, and deduplication."""

from app.ingestion.cleaners.normalizer import TextNormalizer
from app.ingestion.metadata import DocumentDeduplicator


def test_text_normalizer_unicode():
    raw = "Caf\u00e9\t\t\twith   excessive    spaces\r\nand line\r\n\r\n\r\nbreaks."
    cleaned = TextNormalizer.normalize(raw)
    assert "Café with excessive spaces" in cleaned
    assert "\r" not in cleaned
    assert "\t\t" not in cleaned


def test_control_character_removal():
    dirty = "Valid text\x00\x08with null\x0b and bells."
    cleaned = TextNormalizer.normalize(dirty)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Valid text" in cleaned


def test_deterministic_sha256():
    text1 = "Deterministic hashing verification sentence."
    text2 = "Deterministic   hashing   verification   sentence."
    h1 = TextNormalizer.compute_sha256(text1)
    h2 = TextNormalizer.compute_sha256(text2)
    assert h1 == h2
    assert len(h1) == 64


def test_document_deduplicator():
    dedup = DocumentDeduplicator()
    checksum = "abc123hash"
    assert not dedup.is_duplicate_doc(checksum)
    dedup.register_doc(checksum)
    assert dedup.is_duplicate_doc(checksum)
    dedup.remove_doc(checksum)
    assert not dedup.is_duplicate_doc(checksum)
