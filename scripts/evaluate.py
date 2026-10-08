"""Evaluation script executing comparative benchmark against TF-IDF baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.embeddings.model import EmbeddingService
from app.evaluation.benchmarks import RetrievalBenchmarkRunner
from app.evaluation.datasets import (
    generate_benchmark_queries,
    load_benchmark_queries,
    save_benchmark_queries,
)
from app.evaluation.reports import ReportGenerator
from app.indexing.index_manager import IndexManager
from app.retrieval.hybrid import HybridSearchEngine


def run_evaluation(
    index_dir: Path | str = "./data/indexes",
    queries_file: Path | str = "./benchmarks/evaluation_queries.json",
    output_dir: Path | str = "./benchmarks",
    top_k: int = 10,
) -> None:
    console = Console()
    console.print("[bold blue]Starting Information Retrieval Comparative Evaluation...[/bold blue]")

    q_path = Path(queries_file)
    if not q_path.exists():
        console.print(
            f"[yellow]Queries file not found at '{q_path}'. Generating 500 benchmark queries...[/yellow]"
        )
        queries = generate_benchmark_queries(500)
        save_benchmark_queries(queries, q_path)
    else:
        queries = load_benchmark_queries(q_path)

    console.print(
        f"Loaded [bold green]{len(queries)}[/bold green] evaluation queries across 8 categories."
    )

    # Initialize search engine over active index
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

    runner = RetrievalBenchmarkRunner(search_engine=search_engine)
    pipelines = ["tfidf", "semantic", "hybrid", "hybrid_rerank"]
    results = []

    for pipe in pipelines:
        console.print(
            f"Evaluating pipeline mode: [bold cyan]{pipe}[/bold cyan] ({len(queries)} queries)..."
        )
        res = runner.evaluate_pipeline(queries, pipeline_mode=pipe, top_k=top_k)
        results.append(res)
        console.print(
            f"  -> Precision@{top_k}: {res.precision_at_k:.4f}, Recall@{top_k}: {res.recall_at_k:.4f}, "
            f"MRR: {res.mrr_at_k:.4f}, nDCG: {res.ndcg_at_k:.4f}, P50: {res.p50_latency_ms:.1f}ms, P95: {res.p95_latency_ms:.1f}ms"
        )

    # Render rich table
    table = Table(title=f"Retrieval Quality Comparison (Top-{top_k})")
    table.add_column("Pipeline", style="cyan", justify="left")
    table.add_column("Precision@10", justify="center")
    table.add_column("Recall@10", justify="center")
    table.add_column("MRR@10", justify="center")
    table.add_column("nDCG@10", justify="center")
    table.add_column("Hit Rate@10", justify="center")
    table.add_column("P50 Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("P99 Latency", justify="right")

    for r in results:
        table.add_row(
            r.pipeline_name,
            f"{r.precision_at_k:.4f}",
            f"{r.recall_at_k:.4f}",
            f"{r.mrr_at_k:.4f}",
            f"{r.ndcg_at_k:.4f}",
            f"{r.hit_rate_at_k:.4f}",
            f"{r.p50_latency_ms:.1f} ms",
            f"{r.p95_latency_ms:.1f} ms",
            f"{r.p99_latency_ms:.1f} ms",
        )

    console.print(table)

    # Save to disk
    ReportGenerator.save_reports(eval_results=results, load_results=[], output_dir=output_dir)
    console.print(f"[bold green]Saved benchmark reports to '{output_dir}/'[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IR evaluation suite")
    parser.add_argument("--index", type=str, default="./data/indexes", help="Base indexes path")
    parser.add_argument(
        "--queries",
        type=str,
        default="./benchmarks/evaluation_queries.json",
        help="Path to queries JSON",
    )
    parser.add_argument(
        "--output", type=str, default="./benchmarks", help="Output directory for reports"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Evaluation top-K")
    args = parser.parse_args()

    run_evaluation(
        index_dir=args.index,
        queries_file=args.queries,
        output_dir=args.output,
        top_k=args.top_k,
    )
