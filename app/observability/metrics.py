"""Prometheus-compatible metrics instrumentation matching Section 19."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Shared registry
REGISTRY = CollectorRegistry()

# 1. Search request counters & histograms
SEARCH_REQUESTS_TOTAL = Counter(
    "search_requests_total",
    "Total count of search requests",
    ["endpoint", "status", "tenant_id"],
    registry=REGISTRY,
)

SEARCH_LATENCY_SECONDS = Histogram(
    "search_latency_seconds",
    "Total search latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY,
)

EMBEDDING_LATENCY_SECONDS = Histogram(
    "embedding_latency_seconds",
    "Time spent in embedding model inference in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

FAISS_LATENCY_SECONDS = Histogram(
    "faiss_latency_seconds",
    "Time spent in FAISS index vector retrieval in seconds",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=REGISTRY,
)

RERANKER_LATENCY_SECONDS = Histogram(
    "reranker_latency_seconds",
    "Time spent in reranking stage in seconds",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=REGISTRY,
)

# 2. Ingestion counters & gauges
INGESTION_DOCUMENTS_TOTAL = Counter(
    "ingestion_documents_total",
    "Total count of documents ingested",
    ["tenant_id", "status"],
    registry=REGISTRY,
)

INGESTION_FAILURES_TOTAL = Counter(
    "ingestion_failures_total",
    "Total count of document ingestion failures",
    ["reason"],
    registry=REGISTRY,
)

INDEX_SIZE = Gauge(
    "index_size",
    "Current number of vectors in active FAISS index",
    registry=REGISTRY,
)

INDEX_BUILD_SECONDS = Gauge(
    "index_build_seconds",
    "Duration of the latest index build in seconds",
    registry=REGISTRY,
)

API_ERRORS_TOTAL = Counter(
    "api_errors_total",
    "Total count of API error responses",
    ["error_code", "endpoint"],
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    """Generate Prometheus exposition text format."""
    return generate_latest(REGISTRY)
