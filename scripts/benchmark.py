"""Automated latency and high-concurrency load benchmarking script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.embeddings.model import EmbeddingService
from app.evaluation.benchmarks import LoadTestRunner, RetrievalBenchmarkRunner
from app.evaluation.datasets import generate_benchmark_queries, load_benchmark_queries
from app.evaluation.reports import ReportGenerator
from app.indexing.index_manager import IndexManager
from app.retrieval.hybrid import HybridSearchEngine


def run_benchmark(
    concurrency_levels: list[int] | None = None,
    index_dir: Path | str = "./data/indexes",
    queries_file: Path | str = "./benchmarks/evaluation_queries.json",
    output_dir: Path | str = "./benchmarks",
) -> None:
    if concurrency_levels is None:
        concurrency_levels = [10, 50, 100, 250, 500]
    console = Console()
    console.print("[bold blue]Starting Latency & Throughput Benchmark Suite...[/bold blue]")

    q_path = Path(queries_file)
    if q_path.exists():
        queries_objs = load_benchmark_queries(q_path)
        queries = [q.query for q in queries_objs]
    else:
        queries_objs = generate_benchmark_queries(500)
        queries = [q.query for q in queries_objs]

    # Initialize components
    embedding_service = EmbeddingService(dimension=settings.embedding_dimension)
    index_manager = IndexManager(base_dir=Path(index_dir), dimension=settings.embedding_dimension)
    if not index_manager.is_ready():
        console.print(
            "[bold red]Index is not ready! Build index first using 'python scripts/build_index.py'[/bold red]"
        )
        return

    search_engine = HybridSearchEngine(
        embedding_service=embedding_service, index_manager=index_manager
    )
    search_engine.sync_lexical_index()

    # 1. Run IR Evaluation baseline for report
    runner = RetrievalBenchmarkRunner(search_engine=search_engine)
    console.print("Running IR Quality Evaluation across all pipelines...")
    eval_results = [
        runner.evaluate_pipeline(queries_objs[:50], mode, top_k=10)
        for mode in ["tfidf", "semantic", "hybrid", "hybrid_rerank"]
    ]

    # 2. Run High-Concurrency Load Testing
    load_runner = LoadTestRunner(search_engine=search_engine)
    console.print(f"Running Concurrency Stress Benchmark: {concurrency_levels} workers...")
    load_results = load_runner.run_concurrency_benchmark(
        queries=queries,
        concurrency_levels=concurrency_levels,
        requests_per_worker=10,
    )

    # Render load testing table
    table = Table(title="Throughput & Latency Benchmarks by Concurrency")
    table.add_column("Concurrency", justify="center", style="cyan")
    table.add_column("Total Requests", justify="center")
    table.add_column("Success", justify="center")
    table.add_column("Errors", justify="center")
    table.add_column("Throughput (RPS)", justify="right", style="green")
    table.add_column("P50 Latency", justify="right")
    table.add_column("P90 Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("P99 Latency", justify="right")

    for r in load_results:
        table.add_row(
            str(r["concurrency"]),
            str(r["total_requests"]),
            str(r["successful_requests"]),
            str(r["errors"]),
            f"{r['requests_per_second']:.1f}",
            f"{r['p50_ms']:.1f} ms",
            f"{r['p90_ms']:.1f} ms",
            f"{r['p95_ms']:.1f} ms",
            f"{r['p99_ms']:.1f} ms",
        )

    console.print(table)

    # Generate persistent benchmark reports
    ReportGenerator.save_reports(
        eval_results=eval_results, load_results=load_results, output_dir=output_dir
    )
    console.print(
        f"[bold green]Saved benchmark.json, benchmark.csv, and benchmark.md to '{output_dir}/'[/bold green]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run concurrency & latency benchmark")
    parser.add_argument(
        "--concurrency",
        type=str,
        default="10,50,100,250,500",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--index", type=str, default="./data/indexes", help="Base index directory")
    parser.add_argument(
        "--output", type=str, default="./benchmarks", help="Output directory for reports"
    )
    args = parser.parse_args()

    levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip().isdigit()]
    run_benchmark(concurrency_levels=levels, index_dir=args.index, output_dir=args.output)
