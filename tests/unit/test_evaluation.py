"""Unit tests for the information retrieval evaluation and benchmark suite."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from app.evaluation.benchmarks import (
    LoadTestRunner,
    PipelineEvaluationResult,
    RetrievalBenchmarkRunner,
)
from app.evaluation.datasets import (
    BenchmarkQuery,
    generate_benchmark_queries,
    load_benchmark_queries,
    save_benchmark_queries,
)
from app.evaluation.metrics import RetrievalMetrics
from app.evaluation.reports import ReportGenerator


def test_retrieval_metrics_precision():
    # Normal case: 2 out of 5 are relevant
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc99"}
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)

    # Top 2 only
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    # Edge cases
    assert RetrievalMetrics.precision_at_k([], relevant, k=5) == 0.0
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=0) == 0.0
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=-1) == 0.0
    assert RetrievalMetrics.precision_at_k(retrieved, set(), k=5) == 0.0


def test_retrieval_metrics_recall():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc6", "doc7"}
    # 2 hits out of 4 relevant docs
    assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)

    # Top 1 has 1 hit out of 4 relevant docs
    assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.25)

    # Edge cases
    assert RetrievalMetrics.recall_at_k(retrieved, set(), k=5) == 0.0
    assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=0) == 0.0
    assert RetrievalMetrics.recall_at_k([], relevant, k=5) == 0.0


def test_retrieval_metrics_hit_rate():
    retrieved = ["doc1", "doc2", "doc3"]
    assert RetrievalMetrics.hit_rate_at_k(retrieved, {"doc2"}, k=3) == 1.0
    assert RetrievalMetrics.hit_rate_at_k(retrieved, {"doc99"}, k=3) == 0.0
    assert RetrievalMetrics.hit_rate_at_k(retrieved, {"doc3"}, k=2) == 0.0


def test_retrieval_metrics_mrr():
    # Hit at rank 1 -> MRR = 1.0
    assert RetrievalMetrics.mrr_at_k(["doc1", "doc2"], {"doc1"}, k=2) == 1.0
    # Hit at rank 2 -> MRR = 0.5
    assert RetrievalMetrics.mrr_at_k(["doc1", "doc2"], {"doc2"}, k=2) == 0.5
    # Hit at rank 3, but k=2 -> MRR = 0.0
    assert RetrievalMetrics.mrr_at_k(["doc1", "doc2", "doc3"], {"doc3"}, k=2) == 0.0
    # No hits
    assert RetrievalMetrics.mrr_at_k(["doc1", "doc2"], {"doc99"}, k=2) == 0.0


def test_retrieval_metrics_ndcg():
    # Edge cases
    assert RetrievalMetrics.ndcg_at_k([], {"doc1"}, k=5) == 0.0
    assert RetrievalMetrics.ndcg_at_k(["doc1"], set(), k=5) == 0.0

    # Perfect ranking: 2 relevant docs in top 2 positions
    perfect = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2"}
    assert RetrievalMetrics.ndcg_at_k(perfect, relevant, k=3) == pytest.approx(1.0)

    # Imperfect ranking: relevant docs at rank 2 and 3 instead of 1 and 2
    suboptimal = ["doc3", "doc1", "doc2"]
    dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert RetrievalMetrics.ndcg_at_k(suboptimal, relevant, k=3) == pytest.approx(dcg / idcg)


def test_benchmark_datasets(tmp_path):
    # Generation test
    queries = generate_benchmark_queries(target_count=24)
    assert len(queries) >= 24
    assert all(isinstance(q, BenchmarkQuery) for q in queries)
    assert any(q.category == "factual" for q in queries)
    assert any(q.category == "technical" for q in queries)

    # Save and load test
    filepath = tmp_path / "test_queries.json"
    save_benchmark_queries(queries, filepath)
    assert filepath.exists()

    loaded = load_benchmark_queries(filepath)
    assert len(loaded) == len(queries)
    assert loaded[0].query_id == queries[0].query_id
    assert loaded[0].query == queries[0].query
    assert loaded[0].category == queries[0].category
    assert loaded[0].relevant_keywords == queries[0].relevant_keywords


def test_report_generator(tmp_path):
    eval_result = PipelineEvaluationResult(
        pipeline_name="hybrid",
        precision_at_k=0.85,
        recall_at_k=0.75,
        mrr_at_k=0.9,
        ndcg_at_k=0.88,
        hit_rate_at_k=1.0,
        p50_latency_ms=4.2,
        p90_latency_ms=7.1,
        p95_latency_ms=8.5,
        p99_latency_ms=12.0,
        mean_latency_ms=5.0,
        total_queries=10,
    )

    load_result = {
        "concurrency": 10,
        "total_requests": 100,
        "successful_requests": 100,
        "errors": 0,
        "error_rate": 0.0,
        "requests_per_second": 350.5,
        "p50_ms": 2.5,
        "p90_ms": 4.1,
        "p95_ms": 5.0,
        "p99_ms": 8.2,
        "mean_ms": 2.8,
    }

    out_dir = tmp_path / "benchmarks"
    ReportGenerator.save_reports([eval_result], [load_result], output_dir=out_dir)

    assert (out_dir / "benchmark.json").exists()
    assert (out_dir / "benchmark.csv").exists()
    assert (out_dir / "benchmark.md").exists()

    # Validate MD content
    md_text = (out_dir / "benchmark.md").read_text(encoding="utf-8")
    assert "# Production Retrieval & Latency Benchmark Report" in md_text
    assert "**hybrid**" in md_text
    assert "0.8500" in md_text
    assert "350.5" in md_text


def test_retrieval_benchmark_runner():
    mock_search_engine = MagicMock()
    mock_meta_store = MagicMock()
    mock_meta_store.get_all_active_chunks.return_value = [
        {"chunk_id": "c1", "text": "DNS resolves domain names to IP addresses"},
        {"chunk_id": "c2", "text": "HTTPS default port is 443 with TLS encryption"},
    ]
    mock_search_engine.index_manager.metadata_store = mock_meta_store

    mock_tfidf = MagicMock()
    mock_tfidf.search.return_value = [{"chunk_id": "c1", "text": "DNS text", "score": 0.9}]

    mock_search_engine.search.return_value = {
        "results": [
            {"chunk_id": "c2", "text": "HTTPS port 443 text", "score": 0.95},
            {"chunk_id": "c3", "text": "Other unrelated text", "score": 0.4},
        ],
        "latency_ms": 3.5,
    }

    runner = RetrievalBenchmarkRunner(
        search_engine=mock_search_engine, tfidf_retriever=mock_tfidf
    )
    mock_tfidf.fit.assert_called_once()

    queries = [
        BenchmarkQuery(
            query_id="q1",
            query="What is DNS?",
            category="factual",
            relevant_keywords=["dns"],
            relevant_chunk_ids=["c1"],
        ),
        BenchmarkQuery(
            query_id="q2",
            query="HTTPS port",
            category="factual",
            relevant_keywords=["port 443"],
            relevant_chunk_ids=[],
        ),
        BenchmarkQuery(
            query_id="q3",
            query="Unknown query",
            category="other",
            relevant_keywords=[],
            relevant_chunk_ids=[],
        ),
    ]

    # Test tfidf mode
    res_tfidf = runner.evaluate_pipeline(queries, pipeline_mode="tfidf", top_k=5)
    assert res_tfidf.pipeline_name == "tfidf"
    assert res_tfidf.total_queries == 3

    # Test semantic mode
    res_semantic = runner.evaluate_pipeline(queries, pipeline_mode="semantic", top_k=5)
    assert res_semantic.pipeline_name == "semantic"

    # Test hybrid mode
    res_hybrid = runner.evaluate_pipeline(queries, pipeline_mode="hybrid", top_k=5)
    assert res_hybrid.pipeline_name == "hybrid"

    # Test hybrid_rerank mode
    res_rerank = runner.evaluate_pipeline(queries, pipeline_mode="hybrid_rerank", top_k=5)
    assert res_rerank.pipeline_name == "hybrid_rerank"

    # Test invalid mode
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        runner.evaluate_pipeline(queries, pipeline_mode="non_existent")


def test_load_test_runner():
    mock_search_engine = MagicMock()

    # Success case
    mock_search_engine.search.return_value = {"results": [], "latency_ms": 2.0}
    runner = LoadTestRunner(search_engine=mock_search_engine)

    results = runner.run_concurrency_benchmark(
        queries=["test query 1", "test query 2"],
        concurrency_levels=[2, 4],
        requests_per_worker=2,
    )

    assert len(results) == 2
    assert results[0]["concurrency"] == 2
    assert results[0]["total_requests"] == 4
    assert results[0]["successful_requests"] == 4
    assert results[0]["errors"] == 0
    assert results[0]["error_rate"] == 0.0

    # Error case
    mock_search_engine.search.side_effect = RuntimeError("Service unavailable")
    error_results = runner.run_concurrency_benchmark(
        queries=["query"], concurrency_levels=[2], requests_per_worker=1
    )
    assert len(error_results) == 1
    assert error_results[0]["errors"] == 2
    assert error_results[0]["successful_requests"] == 0
    assert error_results[0]["error_rate"] == 1.0
