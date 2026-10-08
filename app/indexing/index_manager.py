"""High-level index coordinator uniting FAISS, SQLite metadata, and versioning."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import IndexNotFoundError
from app.indexing.faiss_index import FaissIndexWrapper
from app.indexing.persistence import MetadataStore
from app.indexing.versioning import IndexVersionManager
from app.ingestion.metadata import ChunkRecord, DocumentRecord

logger = logging.getLogger(__name__)


class IndexManager:
    """Coordinates vector search, metadata filtering, index persistence, and lifecycle."""

    def __init__(
        self,
        base_dir: Path | None = None,
        dimension: int = 512,
        index_type: str = "HNSW",
        metric: str = "cosine",
    ):
        self.base_dir = Path(base_dir or settings.index_dir)
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        self.version_manager = IndexVersionManager(self.base_dir)
        self.faiss_wrapper: FaissIndexWrapper | None = None
        self.metadata_store: MetadataStore | None = None
        self.active_version: str | None = None
        self.lexical_index: Any | None = None  # Loaded BM25 instance if saved

        # Load current index if one exists
        self.load_active_index()

    def load_active_index(self) -> bool:
        """Load the currently active index version from disk, or initialize v001 if none exists."""
        active_dir = self.version_manager.get_active_version_dir()
        if not active_dir or not active_dir.exists():
            logger.info(
                "No active index version found on disk. Initializing initial version v001..."
            )
            return self.initialize_empty_version("v001")

        version_name = active_dir.name
        index_file = active_dir / "index.faiss"
        meta_file = active_dir / "metadata.db"
        bm25_file = active_dir / "bm25.pkl"

        if not (index_file.exists() and meta_file.exists()):
            logger.warning(f"Active index version '{version_name}' is missing required files.")
            return False

        try:
            self.faiss_wrapper = FaissIndexWrapper.load(
                str(index_file),
                ef_search=settings.faiss_ef_search,
                nprobe=settings.faiss_nprobe,
            )
            self.metadata_store = MetadataStore(meta_file)
            self.active_version = version_name

            # Load BM25 if available
            if bm25_file.exists():
                try:
                    with open(bm25_file, "rb") as f:
                        self.lexical_index = pickle.load(f)  # nosec B301
                except Exception:
                    self.lexical_index = None

            logger.info(
                f"Successfully loaded index version '{version_name}' ({self.faiss_wrapper.total_vectors} vectors)."
            )
            return True
        except Exception as e:
            logger.error(f"Error loading active index version '{version_name}': {e}")
            return False

    def initialize_empty_version(self, version_name: str = "v001") -> bool:
        """Create and activate an empty initial index version."""
        version_dir = self.version_manager.get_version_dir(version_name)
        version_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_wrapper = FaissIndexWrapper(
            dimension=self.dimension,
            index_type=self.index_type,
            metric=self.metric,
            ef_search=settings.faiss_ef_search,
            ef_construction=settings.faiss_ef_construction,
            m=settings.faiss_m,
        )
        self.metadata_store = MetadataStore(version_dir / "metadata.db")
        self.faiss_wrapper.save(str(version_dir / "index.faiss"))

        self.version_manager.write_manifest_and_checksum(
            version_dir=version_dir,
            dimension=self.dimension,
            index_type=self.index_type,
            metric=self.metric,
            documents_count=0,
            vectors_count=0,
        )
        self.version_manager.activate_version(version_name, expected_dimension=self.dimension)
        self.active_version = version_name
        return True

    def is_ready(self) -> bool:
        """Verify if index and metadata store are loaded and operational."""
        return self.faiss_wrapper is not None and self.metadata_store is not None

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        tenant_id: str = "default",
        filters: dict[str, Any] | None = None,
        candidate_k: int = 50,
    ) -> list[dict]:
        """Dense semantic search in FAISS with metadata lookup and tenant filtering."""
        if not self.is_ready():
            raise IndexNotFoundError("Index is not loaded or ready for search.")

        assert self.faiss_wrapper is not None
        assert self.metadata_store is not None

        if self.faiss_wrapper.total_vectors == 0:
            return []

        # Retrieve candidates from FAISS
        # Retrieve candidate_k or top_k * 5 to ensure filtering has enough candidates
        k_search = max(top_k * 3, candidate_k)
        scores, faiss_ids = self.faiss_wrapper.search(query_vector, top_k=k_search)

        if len(faiss_ids) == 0 or len(faiss_ids[0]) == 0:
            return []

        retrieved_ids = [int(fid) for fid in faiss_ids[0] if fid != -1]
        score_map = {
            int(fid): float(score)
            for fid, score in zip(faiss_ids[0], scores[0], strict=False)
            if fid != -1
        }

        # Check tenant and optional metadata filter criteria
        matching_ids = self.metadata_store.get_matching_faiss_ids(
            tenant_id=tenant_id, filters=filters
        )
        if matching_ids is not None:
            filtered_ids = [fid for fid in retrieved_ids if fid in matching_ids]
        else:
            filtered_ids = retrieved_ids

        # Fetch chunk details
        chunks_map = self.metadata_store.get_chunks_by_faiss_ids(filtered_ids[:top_k])

        results = []
        for fid in filtered_ids:
            if fid in chunks_map:
                item = chunks_map[fid]
                item["semantic_score"] = score_map.get(fid, 0.0)
                item["score"] = score_map.get(fid, 0.0)
                results.append(item)
                if len(results) >= top_k:
                    break

        return results

    def add_batch(
        self,
        documents: list[DocumentRecord],
        chunks: list[ChunkRecord],
        vectors: np.ndarray,
    ) -> None:
        """Add documents, chunks, and vectors into active index."""
        if not self.is_ready():
            raise IndexNotFoundError("Active index must be initialized before adding documents.")

        assert self.faiss_wrapper is not None
        assert self.metadata_store is not None

        start_id = self.faiss_wrapper.total_vectors
        faiss_ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        # 1. Insert into FAISS
        self.faiss_wrapper.add(vectors, faiss_ids)

        # 2. Insert into SQLite
        self.metadata_store.add_documents(documents)
        chunks_with_ids = list(zip(faiss_ids.tolist(), chunks, strict=False))
        self.metadata_store.add_chunks(chunks_with_ids)

    def delete_document(self, document_id: str, tenant_id: str = "default") -> bool:
        """Tombstone document chunks for immediate exclusion from retrieval."""
        if not self.is_ready():
            raise IndexNotFoundError("Index is not loaded.")
        assert self.metadata_store is not None
        return self.metadata_store.delete_document(document_id, tenant_id=tenant_id)

    def get_status(self) -> dict:
        """Return operational status and metadata of active index."""
        active_version = self.active_version or "None"
        total_vectors = self.faiss_wrapper.total_vectors if self.faiss_wrapper else 0
        total_docs = self.metadata_store.count_documents() if self.metadata_store else 0
        total_chunks = self.metadata_store.count_chunks() if self.metadata_store else 0

        return {
            "status": "ready" if self.is_ready() else "not_loaded",
            "active_version": active_version,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "total_vectors": total_vectors,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "available_versions": self.version_manager.list_version_names(),
        }
