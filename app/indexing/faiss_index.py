"""FAISS Vector Index wrapper supporting HNSW, IVF, and FlatIP."""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np

from app.core.exceptions import DimensionMismatchError, IndexBuildError

logger = logging.getLogger(__name__)


class FaissIndexWrapper:
    """Encapsulates FAISS index construction, tuning, querying, and serialization."""

    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "HNSW",
        metric: str = "cosine",
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        nlist: int = 1024,
        nprobe: int = 32,
    ):
        self.dimension = dimension
        self.index_type = index_type.upper()
        self.metric_type = metric.lower()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.nlist = nlist
        self.nprobe = nprobe

        self.index: faiss.Index | None = None
        self._build_empty_index()

    def _get_faiss_metric(self) -> int:
        if self.metric_type in ("cosine", "ip", "inner_product"):
            return faiss.METRIC_INNER_PRODUCT
        elif self.metric_type in ("l2", "euclidean"):
            return faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {self.metric_type}")

    def _build_empty_index(self) -> None:
        """Construct FAISS index based on configuration."""
        metric = self._get_faiss_metric()

        if self.index_type in ("FLATIP", "FLAT"):
            flat_index: Any = (
                faiss.IndexFlatIP(self.dimension)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(self.dimension)
            )
            self.index = faiss.IndexIDMap2(flat_index)

        elif self.index_type == "HNSW":
            hnsw_index: Any = faiss.IndexHNSWFlat(self.dimension, self.m, metric)
            hnsw_index.hnsw.efConstruction = self.ef_construction
            hnsw_index.hnsw.efSearch = self.ef_search
            self.index = faiss.IndexIDMap2(hnsw_index)

        elif self.index_type == "IVF":
            quantizer: Any = (
                faiss.IndexFlatIP(self.dimension)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(self.dimension)
            )
            ivf_index: Any = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric)
            ivf_index.nprobe = self.nprobe
            self.index = faiss.IndexIDMap2(ivf_index)

        else:
            raise ValueError(f"Unsupported FAISS index_type: {self.index_type}")

    def train_if_needed(self, training_vectors: np.ndarray) -> None:
        """Train IVF index if required."""
        if self.index is None:
            raise IndexBuildError("FAISS index is not initialized.")

        if not self.index.is_trained:
            if training_vectors.shape[0] < self.nlist:
                logger.warning(
                    f"Training vectors count ({training_vectors.shape[0]}) < nlist ({self.nlist}). Adjusting nlist for training."
                )
            self.index.train(training_vectors.astype(np.float32))

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with explicit 64-bit integer IDs."""
        if self.index is None:
            raise IndexBuildError("FAISS index is not initialized.")

        if vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        if not self.index.is_trained:
            self.train_if_needed(vectors)

        vecs_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_i64 = np.ascontiguousarray(ids, dtype=np.int64)

        self.index.add_with_ids(vecs_f32, ids_i64)

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search top_k nearest neighbors. Returns (scores, ids)."""
        if self.index is None or self.index.ntotal == 0:
            return np.empty((query_vectors.shape[0], 0), dtype=np.float32), np.empty(
                (query_vectors.shape[0], 0), dtype=np.int64
            )

        if query_vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(
                f"Query vector dimension {query_vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        # Apply runtime search tuning
        if hasattr(self.index, "index"):
            underlying = self.index.index
            if hasattr(underlying, "hnsw"):
                underlying.hnsw.efSearch = self.ef_search
            if hasattr(underlying, "nprobe"):
                underlying.nprobe = self.nprobe

        k = min(top_k, self.index.ntotal)
        q_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)
        distances, indices = self.index.search(q_f32, k)
        return distances, indices

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def save(self, filepath: str) -> None:
        """Serialize FAISS index to file."""
        if self.index is None:
            raise IndexBuildError("Cannot save uninitialized index.")
        faiss.write_index(self.index, filepath)

    @classmethod
    def load(cls, filepath: str, ef_search: int = 64, nprobe: int = 32) -> FaissIndexWrapper:
        """Load serialized FAISS index from file."""
        raw_index = faiss.read_index(filepath)
        dim = raw_index.d
        wrapper = cls(dimension=dim, ef_search=ef_search, nprobe=nprobe)
        wrapper.index = raw_index
        return wrapper
