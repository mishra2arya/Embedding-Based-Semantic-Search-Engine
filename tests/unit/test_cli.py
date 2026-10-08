"""Unit tests for the Typer command line interface (CLI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from app.cli import cli

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Production-Grade Embedding-Based Semantic Search" in result.stdout
    assert "ingest" in result.stdout
    assert "build-index" in result.stdout
    assert "validate-index" in result.stdout
    assert "search" in result.stdout
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout
    assert "serve" in result.stdout


def test_cli_subcommand_helps():
    for subcmd in [
        "ingest",
        "build-index",
        "validate-index",
        "search",
        "evaluate",
        "benchmark",
        "serve",
    ]:
        result = runner.invoke(cli, [subcmd, "--help"])
        assert result.exit_code == 0


@patch("scripts.ingest.run_ingestion")
def test_cli_ingest(mock_ingest):
    result = runner.invoke(cli, ["ingest", "data/raw", "--output", "data/proc", "--max-docs", "10"])
    assert result.exit_code == 0
    mock_ingest.assert_called_once_with(input_dir="data/raw", output_dir="data/proc", max_docs=10)


@patch("scripts.build_index.build_production_index")
def test_cli_build_index(mock_build):
    result = runner.invoke(
        cli,
        [
            "build-index",
            "--input",
            "data/proc",
            "--output",
            "data/indexes",
            "--index-type",
            "HNSW",
            "--batch-size",
            "128",
            "--max-vectors",
            "100",
        ],
    )
    assert result.exit_code == 0
    mock_build.assert_called_once_with(
        processed_dir="data/proc",
        output_index_dir="data/indexes",
        index_type="HNSW",
        batch_size=128,
        max_vectors=100,
    )


@patch("app.indexing.versioning.IndexVersionManager")
def test_cli_validate_index(mock_vm_cls):
    mock_vm = MagicMock()
    mock_vm_cls.return_value = mock_vm

    # Case 1: No active version
    mock_vm.get_active_version_name.return_value = None
    result = runner.invoke(cli, ["validate-index"])
    assert result.exit_code == 1
    assert "No active index version found" in result.stdout

    # Case 2: Valid version
    mock_vm.get_active_version_name.return_value = "v001"
    mock_vm.validate_version.return_value = True
    result = runner.invoke(cli, ["validate-index"])
    assert result.exit_code == 0
    assert "VALID and production-ready" in result.stdout

    # Case 3: Invalid version
    mock_vm.validate_version.return_value = False
    result = runner.invoke(cli, ["validate-index", "--version", "v002"])
    assert result.exit_code == 1
    assert "CORRUPTED or INVALID" in result.stdout


@patch("app.cli.EmbeddingService")
@patch("app.cli.IndexManager")
@patch("app.cli.HybridSearchEngine")
def test_cli_search(mock_engine_cls, mock_idx_mgr_cls, mock_emb_cls):
    mock_idx_mgr = MagicMock()
    mock_idx_mgr_cls.return_value = mock_idx_mgr

    # Case 1: Index not ready
    mock_idx_mgr.is_ready.return_value = False
    result = runner.invoke(cli, ["search", "query text"])
    assert result.exit_code == 1
    assert "Index is not loaded" in result.stdout

    # Case 2: Index ready, returns search results
    mock_idx_mgr.is_ready.return_value = True
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine
    mock_engine.search.return_value = {
        "latency_ms": 1.25,
        "results": [
            {
                "rank": 1,
                "score": 0.95,
                "title": "Title 1",
                "text": "This is test text for search",
                "citation": "doc_1#c1",
            }
        ],
    }

    result = runner.invoke(cli, ["search", "query text", "--top-k", "1", "--mode", "hybrid"])
    assert result.exit_code == 0
    assert "Title 1" in result.stdout
    assert "Top 1 Search Results" in result.stdout


@patch("scripts.evaluate.run_evaluation")
def test_cli_evaluate(mock_eval):
    result = runner.invoke(cli, ["evaluate", "--index", "data/idx", "--output", "bench", "--top-k", "5"])
    assert result.exit_code == 0
    mock_eval.assert_called_once_with(index_dir="data/idx", output_dir="bench", top_k=5)


@patch("scripts.benchmark.run_benchmark")
def test_cli_benchmark(mock_bm):
    result = runner.invoke(
        cli, ["benchmark", "--concurrency", "10,50", "--index", "data/idx", "--output", "bench"]
    )
    assert result.exit_code == 0
    mock_bm.assert_called_once_with(concurrency_levels=[10, 50], index_dir="data/idx", output_dir="bench")


@patch("uvicorn.run")
def test_cli_serve(mock_uvicorn):
    result = runner.invoke(cli, ["serve", "--host", "127.0.0.1", "--port", "9000", "--workers", "2"])
    assert result.exit_code == 0
    mock_uvicorn.assert_called_once_with("app.main:app", host="127.0.0.1", port=9000, workers=2)
