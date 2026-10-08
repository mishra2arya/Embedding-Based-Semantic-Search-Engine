"""Strategy A: Fixed token-based chunking with sliding window overlap."""

from __future__ import annotations

from app.ingestion.chunking.base import BaseChunker
from app.ingestion.metadata import ChunkRecord


class TokenChunker(BaseChunker):
    """Splits text based on token/word count with configurable overlap."""

    def split(self, text: str, document_id: str, tenant_id: str = "default") -> list[ChunkRecord]:
        words = text.split()
        if not words:
            return []

        # Convert target tokens to target words (~1 word ≈ 1.3 tokens)
        target_words = max(10, int(self.target_tokens / 1.3))
        overlap_words = max(0, int(self.overlap_tokens / 1.3))
        step = max(1, target_words - overlap_words)

        chunks: list[ChunkRecord] = []
        chunk_idx = 0

        for i in range(0, len(words), step):
            window = words[i : i + target_words]
            chunk_text = " ".join(window)
            if chunk_text.strip():
                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        tenant_id=tenant_id,
                        extra_metadata={"start_word": i, "end_word": i + len(window)},
                    )
                )
                chunk_idx += 1

            if i + target_words >= len(words):
                break

        return chunks
