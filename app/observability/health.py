"""Health check and readiness verification services."""

from __future__ import annotations

import datetime
import os
from typing import Any

import psutil

from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager

START_TIME = datetime.datetime.now(datetime.UTC)


class HealthChecker:
    """Evaluates liveness and readiness of application components."""

    def __init__(self, embedding_service: EmbeddingService, index_manager: IndexManager):
        self.embedding_service = embedding_service
        self.index_manager = index_manager

    def check_liveness(self) -> dict[str, Any]:
        """Verify that the API process is alive and responsive."""
        uptime = (datetime.datetime.now(datetime.UTC) - START_TIME).total_seconds()
        process = psutil.Process(os.getpid())

        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "process": {
                "pid": os.getpid(),
                "memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "cpu_percent": process.cpu_percent(),
            },
        }

    def check_readiness(self) -> dict[str, Any]:
        """Verify that model is loaded, index is loaded, and metadata store is operational."""
        model_ready = self.embedding_service is not None
        index_ready = self.index_manager.is_ready()
        meta_ready = (
            self.index_manager.metadata_store is not None
            and self.index_manager.metadata_store.count_documents() >= 0
        )

        overall_ready = model_ready and index_ready and meta_ready

        return {
            "ready": overall_ready,
            "status": "ready" if overall_ready else "not_ready",
            "checks": {
                "embedding_model": "loaded" if model_ready else "unavailable",
                "vector_index": "loaded" if index_ready else "unavailable",
                "metadata_store": "connected" if meta_ready else "unavailable",
            },
            "index_status": self.index_manager.get_status() if index_ready else None,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
