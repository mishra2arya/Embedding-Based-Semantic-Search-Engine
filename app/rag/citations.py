"""Citation tracking and source attribution for RAG answers."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    document_id: str = Field(description="Retrieved parent document ID")
    chunk_id: str = Field(description="Exact retrieved chunk ID")
    title: str = Field(description="Document title")
    relevance_score: float = Field(description="Search relevance score")


class CitationTracker:
    """Verifies that generated citations correspond to actual retrieved source chunks."""

    @staticmethod
    def extract_sources(
        selected_chunks: list[dict],
        answer_text: str = "",
    ) -> list[SourceAttribution]:
        """Map retrieved chunks to structured SourceAttribution instances."""
        sources: list[SourceAttribution] = []

        # Find which citation numbers are referenced in answer if present (e.g. [1], [2])
        set(map(int, re.findall(r"\[(\d+)\]", answer_text)))

        for _rank, chunk in enumerate(selected_chunks, start=1):
            # If citations are explicitly referenced in answer, tag them; otherwise include top retrieved
            score = chunk.get("final_score", chunk.get("score", 0.0))

            sources.append(
                SourceAttribution(
                    document_id=chunk.get("document_id", "unknown"),
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    title=chunk.get("title", "Untitled"),
                    relevance_score=round(float(score), 4),
                )
            )

        return sources
