"""Strategy B: Sentence-aware chunking preserving grammatical boundaries."""

from __future__ import annotations

import re

from app.ingestion.chunking.base import BaseChunker
from app.ingestion.metadata import ChunkRecord


class SentenceChunker(BaseChunker):
    """Splits around sentence boundaries and groups coherent sentence blocks."""

    # Sentence boundary regex with common abbreviations protection
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\"'“])")

    def _split_sentences(self, text: str) -> list[str]:
        raw_sentences = self._SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def split(self, text: str, document_id: str, tenant_id: str = "default") -> list[ChunkRecord]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[ChunkRecord] = []
        current_sentences: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for sentence in sentences:
            s_tokens = self.estimate_tokens(sentence)

            # If adding this sentence exceeds max_tokens and we already have at least min_tokens
            if current_tokens + s_tokens > self.target_tokens and current_tokens >= self.min_tokens:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        tenant_id=tenant_id,
                        extra_metadata={"sentence_count": len(current_sentences)},
                    )
                )
                chunk_idx += 1

                # Carry over last sentence for overlap if overlap_tokens > 0
                if self.overlap_tokens > 0 and current_sentences:
                    current_sentences = [current_sentences[-1], sentence]
                    current_tokens = self.estimate_tokens(" ".join(current_sentences))
                else:
                    current_sentences = [sentence]
                    current_tokens = s_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += s_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self.create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    tenant_id=tenant_id,
                    extra_metadata={"sentence_count": len(current_sentences)},
                )
            )

        return chunks
