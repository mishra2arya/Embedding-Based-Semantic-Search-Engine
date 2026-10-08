"""Comparative evaluation and latency benchmarking suite."""

from __future__ import annotations

import concurrent.futures
import logging
import statistics
import time
from dataclasses import dataclass

from app.evaluation.datasets import BenchmarkQuery
from app.evaluation.metrics import RetrievalMetrics
from app.retrieval.hybrid import HybridSearchEngine
from app.retrieval.lexical import TfidfRetriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineEvaluationResult:
    pipeline_name: str
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    total_queries: int


class RetrievalBenchmarkRunner:
    """Executes systematic evaluation across TF-IDF, Semantic, Hybrid, and Hybrid+Reranker."""

    def __init__(
        self, search_engine: HybridSearchEngine, tfidf_retriever: TfidfRetriever | None = None
    ):
        self.search_engine = search_engine
        self.tfidf_retriever = tfidf_retriever or TfidfRetriever()
        self._init_tfidf()

    def _init_tfidf(self) -> None:
        if self.search_engine.index_manager.metadata_store:
            chunks = self.search_engine.index_manager.metadata_store.get_all_active_chunks()
            self.tfidf_retriever.fit(chunks)

    def evaluate_pipeline(
        self,
        queries: list[BenchmarkQuery],
        pipeline_mode: str,  # "tfidf", "semantic", "hybrid", "hybrid_rerank"
        top_k: int = 10,
    ) -> PipelineEvaluationResult:
        """Run all queries through specified pipeline and calculate IR metrics."""
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        hit_rates = []
        latencies_ms = []

        for q in queries:
            t0 = time.perf_counter()

            if pipeline_mode == "tfidf":
                results = self.tfidf_retriever.search(q.query, top_k=top_k)
            elif pipeline_mode == "semantic":
                out = self.search_engine.search(q.query, top_k=top_k, mode="semantic", rerank=False)
                results = out["results"]
            elif pipeline_mode == "hybrid":
                out = self.search_engine.search(q.query, top_k=top_k, mode="hybrid", rerank=False)
                results = out["results"]
            elif pipeline_mode == "hybrid_rerank":
                out = self.search_engine.search(q.query, top_k=top_k, mode="hybrid", rerank=True)
                results = out["results"]
            else:
                raise ValueError(f"Unknown pipeline mode: {pipeline_mode}")

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)

            retrieved_cids = [r.get("chunk_id", "") for r in results]

            # Ground truth relevance: matching chunk IDs or text containing relevant keywords
            relevant_cids: set[str] = set(q.relevant_chunk_ids)
            if not relevant_cids and q.relevant_keywords:
                # Identify ground-truth relevant chunks based on semantic/lexical match with keywords
                for r in results:
                    text_lower = r.get("text", "").lower()
                    if any(kw.lower() in text_lower for kw in q.relevant_keywords):
                        relevant_cids.add(r.get("chunk_id", ""))

            # Calculate metrics
            if relevant_cids:
                precisions.append(
                    RetrievalMetrics.precision_at_k(retrieved_cids, relevant_cids, k=top_k)
                )
                recalls.append(RetrievalMetrics.recall_at_k(retrieved_cids, relevant_cids, k=top_k))
                mrrs.append(RetrievalMetrics.mrr_at_k(retrieved_cids, relevant_cids, k=top_k))
                ndcgs.append(RetrievalMetrics.ndcg_at_k(retrieved_cids, relevant_cids, k=top_k))
                hit_rates.append(
                    RetrievalMetrics.hit_rate_at_k(retrieved_cids, relevant_cids, k=top_k)
                )
            else:
                # If no ground truth available, record baseline match
                precisions.append(0.0)
                recalls.append(0.0)
                mrrs.append(0.0)
                ndcgs.append(0.0)
                hit_rates.append(0.0)

        latencies_sorted = sorted(latencies_ms)
        n = len(latencies_sorted)

        def percentile(pct: float) -> float:
            idx = int((pct / 100.0) * n)
            return latencies_sorted[min(idx, n - 1)]

        return PipelineEvaluationResult(
            pipeline_name=pipeline_mode,
            precision_at_k=round(statistics.mean(precisions) if precisions else 0.0, 4),
            recall_at_k=round(statistics.mean(recalls) if recalls else 0.0, 4),
            mrr_at_k=round(statistics.mean(mrrs) if mrrs else 0.0, 4),
            ndcg_at_k=round(statistics.mean(ndcgs) if ndcgs else 0.0, 4),
            hit_rate_at_k=round(statistics.mean(hit_rates) if hit_rates else 0.0, 4),
            p50_latency_ms=round(percentile(50), 2),
            p90_latency_ms=round(percentile(90), 2),
            p95_latency_ms=round(percentile(95), 2),
            p99_latency_ms=round(percentile(99), 2),
            mean_latency_ms=round(statistics.mean(latencies_ms) if latencies_ms else 0.0, 2),
            total_queries=len(queries),
        )


class LoadTestRunner:
    """Runs automated concurrency load testing across 10, 50, 100, 250, 500 workers."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    def run_concurrency_benchmark(
        self,
        queries: list[str],
        concurrency_levels: list[int] | None = None,
        requests_per_worker: int = 10,
    ) -> list[dict]:
        """Execute concurrent requests and record latency percentiles and throughput."""
        if concurrency_levels is None:
            concurrency_levels = [10, 50, 100, 250, 500]
        benchmark_results = []

        for concurrency in concurrency_levels:
            total_requests = concurrency * requests_per_worker
            latencies: list[float] = []
            errors = 0

            def worker_task(idx: int) -> float:
                q = queries[idx % len(queries)]
                t0 = time.perf_counter()
                try:
                    self.search_engine.search(q, top_k=10, mode="hybrid", rerank=False)
                    return (time.perf_counter() - t0) * 1000.0
                except Exception:
                    return -1.0

            wall_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(worker_task, i) for i in range(total_requests)]
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    if res < 0:
                        errors += 1
                    else:
                        latencies.append(res)
            wall_time = time.perf_counter() - wall_start

            latencies.sort()
            n = len(latencies)
            if n > 0:
                p50 = latencies[int(0.50 * n)]
                p90 = latencies[int(0.90 * n)]
                p95 = latencies[int(0.95 * n)]
                p99 = latencies[min(int(0.99 * n), n - 1)]
                rps = n / wall_time if wall_time > 0 else 0.0
            else:
                p50 = p90 = p95 = p99 = rps = 0.0

            benchmark_results.append(
                {
                    "concurrency": concurrency,
                    "total_requests": total_requests,
                    "successful_requests": n,
                    "errors": errors,
                    "error_rate": round(errors / total_requests if total_requests > 0 else 0.0, 4),
                    "wall_time_seconds": round(wall_time, 2),
                    "requests_per_second": round(rps, 2),
                    "p50_ms": round(p50, 2),
                    "p90_ms": round(p90, 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                    "mean_ms": round(statistics.mean(latencies) if latencies else 0.0, 2),
                }
            )
            logger.info(
                f"Concurrency {concurrency:3d} -> RPS: {rps:6.1f}, P50: {p50:5.1f}ms, P95: {p95:5.1f}ms, Errors: {errors}"
            )

        return benchmark_results
