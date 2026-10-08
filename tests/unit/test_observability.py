"""Unit tests for request profiler and semantic retriever."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import IndexNotFoundError
from app.observability.tracing import RequestProfiler
from app.retrieval.semantic import SemanticRetriever


def test_request_profiler():
    profiler = RequestProfiler()
    with profiler.time_step("encoding"):
        time.sleep(0.01)

    with profiler.time_step("search"):
        time.sleep(0.01)

    summary = profiler.get_summary()
    assert "encoding" in summary
    assert "search" in summary
    assert "total_ms" in summary
    assert summary["encoding"] > 0
    assert summary["search"] > 0
    assert summary["total_ms"] >= summary["encoding"] + summary["search"]


def test_semantic_retriever():
    mock_emb = MagicMock()
    mock_idx = MagicMock()

    retriever = SemanticRetriever(embedding_service=mock_emb, index_manager=mock_idx)

    # When index not ready -> raises IndexNotFoundError
    mock_idx.is_ready.return_value = False
    with pytest.raises(IndexNotFoundError):
        retriever.search("test query")

    # When index is ready -> executes search
    mock_idx.is_ready.return_value = True
    mock_emb.encode.return_value = [0.1] * 512
    mock_idx.search.return_value = [{"chunk_id": "c1", "score": 0.9}]

    results = retriever.search("test query", top_k=5, tenant_id="tenant_1", filters={"cat": "ai"})
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    mock_emb.encode.assert_called_once_with("test query", normalize=True, use_cache=True)
    mock_idx.search.assert_called_once()
