"""Semantic dense vector retrieval querying FAISS."""

from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import IndexNotFoundError
from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Dense retriever using query embeddings and FAISS index search."""

    def __init__(self, embedding_service: EmbeddingService, index_manager: IndexManager):
        self.embedding_service = embedding_service
        self.index_manager = index_manager

    def search(
        self,
        query: str,
        top_k: int = 50,
        tenant_id: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Encode query and perform vector nearest-neighbor search."""
        if not self.index_manager.is_ready():
            raise IndexNotFoundError("Vector index is not initialized.")

        query_vec = self.embedding_service.encode(query, normalize=True, use_cache=True)
        results = self.index_manager.search(
            query_vector=query_vec,
            top_k=top_k,
            tenant_id=tenant_id,
            filters=filters,
            candidate_k=top_k,
        )
        return results
