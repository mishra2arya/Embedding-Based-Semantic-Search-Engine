"""Unit tests for second-stage rerankers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.retrieval.reranking import CrossEncoderReranker, FastAlignmentReranker


def test_fast_alignment_reranker():
    reranker = FastAlignmentReranker()

    # Empty candidates
    assert reranker.rerank("test query", []) == []

    candidates = [
        {
            "chunk_id": "c1",
            "score": 0.5,
            "title": "Random doc",
            "text": "This is completely irrelevant text.",
            "metadata": {},
        },
        {
            "chunk_id": "c2",
            "score": 0.5,
            "title": "Vector databases guide",
            "text": "Faiss enables fast similarity search on dense vectors.",
            "metadata": {"section_heading": "Vector Indexing"},
        },
        {
            "chunk_id": "c3",
            "score": 0.5,
            "title": "Exact match doc",
            "text": "Here is faiss similarity search exactly in text.",
            "metadata": {},
        },
    ]

    ranked = reranker.rerank("faiss similarity search", candidates, top_k=2)
    assert len(ranked) == 2
    # c3 has exact phrase match, should be ranked top
    assert ranked[0]["chunk_id"] == "c3"
    assert "rerank_score" in ranked[0]
    assert "final_score" in ranked[0]


def test_cross_encoder_reranker_fallback():
    with patch.object(CrossEncoderReranker, "_init_model"):
        reranker = CrossEncoderReranker()
        reranker._model = None

        candidates = [
            {"chunk_id": "c1", "score": 0.3, "text": "Unrelated text", "title": "A"},
            {"chunk_id": "c2", "score": 0.4, "text": "Kubernetes Ingress routing", "title": "K8s"},
        ]

        res = reranker.rerank("Kubernetes Ingress", candidates, top_k=2)
        assert len(res) == 2
        assert res[0]["chunk_id"] == "c2"

        # Empty candidates
        assert reranker.rerank("query", []) == []


def test_cross_encoder_reranker_with_mock_model():
    with patch.object(CrossEncoderReranker, "_init_model"):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9]
        reranker._model = mock_model

        candidates = [
            {"chunk_id": "c1", "score": 0.5, "text": "First doc", "title": "Doc 1"},
            {"chunk_id": "c2", "score": 0.5, "text": "Second doc", "title": "Doc 2"},
        ]

        ranked = reranker.rerank("query", candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["chunk_id"] == "c2"
        assert ranked[0]["rerank_score"] == 0.9

        # Test predict exception fallback with fresh candidate dicts
        mock_model.predict.side_effect = RuntimeError("Prediction error")
        fresh_candidates = [
            {"chunk_id": "c1", "score": 0.5, "text": "Target doc phrase", "title": "Doc 1"},
            {"chunk_id": "c2", "score": 0.5, "text": "Completely unrelated content", "title": "Doc 2"},
        ]
        fallback_ranked = reranker.rerank("Target doc phrase", fresh_candidates, top_k=2)
        assert len(fallback_ranked) == 2
        assert fallback_ranked[0]["chunk_id"] == "c1"


def test_cross_encoder_init_failure():
    with patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("Model init failure")):
        reranker = CrossEncoderReranker()
        assert reranker._model is None
