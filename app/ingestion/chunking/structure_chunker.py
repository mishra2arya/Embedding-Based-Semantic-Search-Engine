"""Strategy D: Structure-aware chunking respecting headings, paragraphs, lists, and code blocks."""

from __future__ import annotations

import re

from app.ingestion.chunking.base import BaseChunker
from app.ingestion.metadata import ChunkRecord


class StructureChunker(BaseChunker):
    """Splits structured text respecting headers, code blocks, tables, and paragraphs."""

    # Regex patterns for structural delimiters
    _HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z0-9\s]{3,}\n[-=]{3,})$", re.MULTILINE)
    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)")

    def _split_into_structural_blocks(self, text: str) -> list[str]:
        """Separate code blocks and paragraphs into atomic structural units."""
        # Split on code fences first to preserve code blocks intact
        parts = self._CODE_BLOCK_RE.split(text)
        blocks: list[str] = []

        for part in parts:
            if not part.strip():
                continue
            if part.startswith("```") and part.endswith("```"):
                blocks.append(part)
            else:
                # Split regular text on double newlines or headings
                subparts = re.split(r"\n\s*\n", part)
                for sp in subparts:
                    sp = sp.strip()
                    if sp:
                        blocks.append(sp)

        return blocks

    def split(self, text: str, document_id: str, tenant_id: str = "default") -> list[ChunkRecord]:
        blocks = self._split_into_structural_blocks(text)
        if not blocks:
            return []

        chunks: list[ChunkRecord] = []
        current_blocks: list[str] = []
        current_tokens = 0
        chunk_idx = 0
        current_heading: str | None = None

        for block in blocks:
            # Check if block is a heading
            if self._HEADING_RE.match(block):
                current_heading = block.strip()

            block_tokens = self.estimate_tokens(block)

            # If adding this block exceeds target_tokens and we meet min_tokens
            if (
                current_tokens + block_tokens > self.target_tokens
                and current_tokens >= self.min_tokens
            ):
                chunk_text = "\n\n".join(current_blocks)
                meta = {}
                if current_heading:
                    meta["section_heading"] = current_heading

                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        tenant_id=tenant_id,
                        extra_metadata=meta,
                    )
                )
                chunk_idx += 1
                current_blocks = [block]
                current_tokens = block_tokens
            else:
                current_blocks.append(block)
                current_tokens += block_tokens

        if current_blocks:
            chunk_text = "\n\n".join(current_blocks)
            meta = {}
            if current_heading:
                meta["section_heading"] = current_heading

            chunks.append(
                self.create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    tenant_id=tenant_id,
                    extra_metadata=meta,
                )
            )

        return chunks
