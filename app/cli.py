"""Command Line Interface (CLI) for Semantic Search Engine matching Section 36."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager
from app.retrieval.hybrid import HybridSearchEngine

cli = typer.Typer(
    name="semantic-search",
    help="Production-Grade Embedding-Based Semantic Search & RAG Platform CLI",
    add_completion=False,
)
console = Console()


@cli.command("ingest")
def ingest_cmd(
    input_path: str = typer.Argument("./data/raw", help="Path to raw document directory or file"),
    output_path: str = typer.Option(
        "./data/processed", "--output", "-o", help="Output directory for processed chunks"
    ),
    max_docs: int = typer.Option(0, "--max-docs", "-m", help="Max documents to ingest (0 = all)"),
):
    """Ingest, parse, normalize, and chunk document files."""
    from scripts.ingest import run_ingestion

    run_ingestion(input_dir=input_path, output_dir=output_path, max_docs=max_docs)


@cli.command("build-index")
def build_index_cmd(
    input_path: str = typer.Option(
        "./data/processed", "--input", "-i", help="Directory of processed chunks"
    ),
    output_path: str = typer.Option(
        "./data/indexes", "--output", "-o", help="Index storage directory"
    ),
    index_type: str = typer.Option(
        "HNSW", "--index-type", "-t", help="FAISS index type (HNSW, IVF, FlatIP)"
    ),
    batch_size: int = typer.Option(256, "--batch-size", "-b", help="Vector batch size"),
    max_vectors: int = typer.Option(0, "--max-vectors", help="Max vectors to index (0 = all)"),
):
    """Build and activate a new FAISS and metadata index version."""
    from scripts.build_index import build_production_index

    build_production_index(
        processed_dir=input_path,
        output_index_dir=output_path,
        index_type=index_type,
        batch_size=batch_size,
        max_vectors=max_vectors,
    )


@cli.command("validate-index")
def validate_index_cmd(
    index_path: str = typer.Option("./data/indexes", "--index-path", help="Base index directory"),
    version: str | None = typer.Option(
        None, "--version", help="Specific version to validate (default: active)"
    ),
):
    """Validate index integrity, file checksums, and searchability."""
    from app.indexing.versioning import IndexVersionManager

    vm = IndexVersionManager(index_path)
    target_version = version or vm.get_active_version_name()

    if not target_version:
        console.print("[bold red]No active index version found to validate.[/bold red]")
        raise typer.Exit(code=1)

    v_dir = vm.get_version_dir(target_version)
    is_valid = vm.validate_version(v_dir, expected_dimension=settings.embedding_dimension)

    if is_valid:
        console.print(
            f"[bold green]Version '{target_version}' is VALID and production-ready.[/bold green]"
        )
    else:
        console.print(f"[bold red]Version '{target_version}' is CORRUPTED or INVALID.[/bold red]")
        raise typer.Exit(code=1)


@cli.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: hybrid, semantic, lexical"
    ),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Enable reranking"),
    tenant_id: str = typer.Option("default", "--tenant", help="Tenant workspace ID"),
):
    """Execute search query from terminal with score breakdown."""
    emb = EmbeddingService(dimension=settings.embedding_dimension)
    idx_mgr = IndexManager(base_dir=settings.index_dir, dimension=settings.embedding_dimension)
    if not idx_mgr.is_ready():
        console.print("[bold red]Index is not loaded. Build index first.[/bold red]")
        raise typer.Exit(code=1)

    engine = HybridSearchEngine(embedding_service=emb, index_manager=idx_mgr)
    engine.sync_lexical_index(tenant_id=tenant_id)

    out = engine.search(query=query, top_k=top_k, mode=mode, rerank=rerank, tenant_id=tenant_id)

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(
        f"[bold]Latency:[/bold] {out['latency_ms']:.2f} ms | [bold]Mode:[/bold] {mode} | [bold]Reranked:[/bold] {rerank}\n"
    )

    table = Table(title=f"Top {len(out['results'])} Search Results")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Title", style="bold")
    table.add_column("Matched Chunk Preview", style="dim")
    table.add_column("Citation", style="yellow")

    for r in out["results"]:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.4f}",
            r["title"][:30],
            r["text"][:60] + "...",
            r["citation"],
        )

    console.print(table)


@cli.command("evaluate")
def evaluate_cmd(
    index_path: str = typer.Option("./data/indexes", "--index", help="Base indexes path"),
    output_dir: str = typer.Option("./benchmarks", "--output", help="Output directory for reports"),
    top_k: int = typer.Option(10, "--top-k", help="Top-K metric evaluation"),
):
    """Run full comparative IR evaluation across TF-IDF, Semantic, Hybrid, and Reranker."""
    from scripts.evaluate import run_evaluation

    run_evaluation(index_dir=index_path, output_dir=output_dir, top_k=top_k)


@cli.command("benchmark")
def benchmark_cmd(
    concurrency: str = typer.Option(
        "10,50,100,250,500", "--concurrency", help="Comma-separated concurrency workers"
    ),
    index_path: str = typer.Option("./data/indexes", "--index", help="Base indexes path"),
    output_dir: str = typer.Option("./benchmarks", "--output", help="Output directory for reports"),
):
    """Run automated load and throughput benchmark."""
    from scripts.benchmark import run_benchmark

    levels = [int(c.strip()) for c in concurrency.split(",") if c.strip().isdigit()]
    run_benchmark(concurrency_levels=levels, index_dir=index_path, output_dir=output_dir)


@cli.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),  # nosec B104
    port: int = typer.Option(8000, "--port", help="Bind port"),
    workers: int = typer.Option(1, "--workers", help="Worker processes"),
):
    """Start FastAPI REST server and web dashboard."""
    console.print(
        f"[bold green]Starting Semantic Search Engine on http://{host}:{port}...[/bold green]"
    )
    uvicorn.run("app.main:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    cli()
