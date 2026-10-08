"""Chunker factory for dynamic strategy instantiation."""

from __future__ import annotations

from app.ingestion.chunking.base import BaseChunker
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.chunking.sentence_chunker import SentenceChunker
from app.ingestion.chunking.structure_chunker import StructureChunker
from app.ingestion.chunking.token_chunker import TokenChunker


class ChunkerFactory:
    """Factory to retrieve configured chunking strategy."""

    _STRATEGIES: dict[str, type[BaseChunker]] = {
        "token": TokenChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
        "structure": StructureChunker,
    }

    @classmethod
    def get_chunker(
        cls,
        strategy: str = "structure",
        target_tokens: int = 350,
        min_tokens: int = 100,
        max_tokens: int = 600,
        overlap_tokens: int = 50,
        **kwargs,
    ) -> BaseChunker:
        strat_key = strategy.lower()
        chunker_cls = cls._STRATEGIES.get(strat_key, StructureChunker)
        return chunker_cls(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            **kwargs,
        )
