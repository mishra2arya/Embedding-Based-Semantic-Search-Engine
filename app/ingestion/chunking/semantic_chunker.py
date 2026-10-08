"""Strategy C: Semantic chunking grouping adjacent sentences by embedding similarity."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from app.ingestion.chunking.base import BaseChunker
from app.ingestion.metadata import ChunkRecord


class SemanticChunker(BaseChunker):
    """Groups adjacent sentences according to semantic / embedding similarity."""

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\"'“])")

    def __init__(
        self,
        target_tokens: int = 350,
        min_tokens: int = 100,
        max_tokens: int = 600,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.65,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ):
        super().__init__(target_tokens, min_tokens, max_tokens, overlap_tokens)
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn

    def _split_sentences(self, text: str) -> list[str]:
        raw = self._SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in raw if s.strip()]

    def _compute_fallback_vectors(self, sentences: list[str]) -> np.ndarray:
        """Compute fast TF-IDF-like token frequency vectors when dense model is omitted."""
        vocab: dict[str, int] = {}
        for s in sentences:
            for word in s.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        if not vocab:
            return np.ones((len(sentences), 1), dtype=np.float32)

        mat = np.zeros((len(sentences), len(vocab)), dtype=np.float32)
        for i, s in enumerate(sentences):
            for word in s.lower().split():
                if word in vocab:
                    mat[i, vocab[word]] += 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def split(self, text: str, document_id: str, tenant_id: str = "default") -> list[ChunkRecord]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                self.create_chunk(
                    text=sentences[0],
                    document_id=document_id,
                    chunk_index=0,
                    tenant_id=tenant_id,
                )
            ]

        # Get sentence representations
        if self.embed_fn:
            vectors = self.embed_fn(sentences)
        else:
            vectors = self._compute_fallback_vectors(sentences)

        # Compute cosine similarity between adjacent sentences
        similarities: list[float] = []
        for i in range(len(sentences) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            similarities.append(sim)

        # Determine breakpoints
        chunks: list[ChunkRecord] = []
        current_sentences: list[str] = [sentences[0]]
        current_tokens = self.estimate_tokens(sentences[0])
        chunk_idx = 0

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_tokens = self.estimate_tokens(next_sentence)

            # Break if semantic shift detected and size is adequate, OR if max_tokens reached
            is_semantic_break = (
                sim < self.similarity_threshold and current_tokens >= self.min_tokens
            )
            is_size_break = current_tokens + next_tokens > self.max_tokens

            if is_semantic_break or is_size_break:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        tenant_id=tenant_id,
                        extra_metadata={"semantic_split": is_semantic_break, "similarity": sim},
                    )
                )
                chunk_idx += 1
                current_sentences = [next_sentence]
                current_tokens = next_tokens
            else:
                current_sentences.append(next_sentence)
                current_tokens += next_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self.create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    tenant_id=tenant_id,
                )
            )

        return chunks
