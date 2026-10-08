"""Text normalization, sanitation, and content hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata


class TextNormalizer:
    """Standardizes text for deduplication, indexing, and tokenization."""

    # Pre-compile regex for performance
    _WHITESPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    @classmethod
    def normalize(cls, text: str) -> str:
        """Perform deterministic Unicode NFKC normalization and whitespace cleanup."""
        if not text:
            return ""

        # Remove control characters
        text = cls._CONTROL_CHAR_RE.sub("", text)

        # Unicode NFKC normalization
        text = unicodedata.normalize("NFKC", text)

        # Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal whitespace
        text = cls._WHITESPACE_RE.sub(" ", text)

        # Collapse excessive newlines (max 2 consecutive newlines)
        text = cls._MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()

    @classmethod
    def compute_sha256(cls, text: str) -> str:
        """Compute deterministic SHA-256 hash of normalized text."""
        normalized = cls.normalize(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
