"""Admin and index management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_current_user,
    get_index_manager,
    get_search_engine,
    require_admin,
)
from app.core.security import Role
from app.indexing.index_manager import IndexManager
from app.retrieval.hybrid import HybridSearchEngine

router = APIRouter(prefix="/index", tags=["Admin & Index Management"])


class IndexStatusResponse(BaseModel):
    status: str
    active_version: str
    dimension: int
    index_type: str
    metric: str
    total_vectors: int
    total_documents: int
    total_chunks: int
    available_versions: list[str]


class ActivateVersionRequest(BaseModel):
    version_name: str = Field(description="Target version name to activate, e.g. v001")


@router.get("/status", response_model=IndexStatusResponse, summary="Get index status")
@router.get("", response_model=IndexStatusResponse, summary="Get index status (alias)")
def get_index_status(
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
):
    """Retrieve operational status, active version, dimension, and vector counts."""
    return index_mgr.get_status()


@router.post("/activate", summary="Activate specific index version")
def activate_version(
    req: ActivateVersionRequest,
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
    engine: HybridSearchEngine = Depends(get_search_engine),
    _: None = Depends(require_admin),
):
    """Validate and atomically activate target index version."""
    index_mgr.version_manager.activate_version(
        req.version_name, expected_dimension=index_mgr.dimension
    )
    index_mgr.load_active_index()
    engine.sync_lexical_index()
    return {"status": "activated", "active_version": req.version_name}


@router.post("/rollback", summary="Rollback to previous index version")
def rollback_version(
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
    engine: HybridSearchEngine = Depends(get_search_engine),
    _: None = Depends(require_admin),
):
    """Rollback active index to immediately prior valid version."""
    prev_ver = index_mgr.version_manager.rollback_version(expected_dimension=index_mgr.dimension)
    index_mgr.load_active_index()
    engine.sync_lexical_index()
    return {"status": "rolled_back", "active_version": prev_ver}


@router.post("/rebuild", summary="Compact and rebuild active index")
def rebuild_index(
    user: tuple[str, Role] = Depends(get_current_user),
    index_mgr: IndexManager = Depends(get_index_manager),
    engine: HybridSearchEngine = Depends(get_search_engine),
    _: None = Depends(require_admin),
):
    """Rebuild and compact index, permanently purging tombstoned chunks."""
    if not index_mgr.is_ready() or not index_mgr.metadata_store:
        return {"status": "error", "message": "Index not loaded."}

    # Fetch active non-tombstoned chunks
    active_chunks = index_mgr.metadata_store.get_all_active_chunks()
    if not active_chunks:
        return {"status": "empty", "message": "No active chunks to rebuild."}

    # Generate new version directory
    new_ver_name = index_mgr.version_manager.get_next_version_name()
    new_ver_dir = index_mgr.version_manager.get_version_dir(new_ver_name)
    new_ver_dir.mkdir(parents=True, exist_ok=True)

    from app.indexing.faiss_index import FaissIndexWrapper
    from app.indexing.persistence import MetadataStore

    new_faiss = FaissIndexWrapper(
        dimension=index_mgr.dimension,
        index_type=index_mgr.index_type,
        metric=index_mgr.metric,
    )
    new_meta = MetadataStore(new_ver_dir / "metadata.db")

    # Re-embed active chunks and populate new version
    texts = [c["text"] for c in active_chunks]
    vectors = engine.embedding_service.encode(texts, normalize=True, use_cache=False)

    import numpy as np

    from app.ingestion.metadata import ChunkRecord

    rebuilt_chunks = [
        ChunkRecord(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            chunk_index=i,
            text=c["text"],
            checksum=str(hash(c["text"])),
        )
        for i, c in enumerate(active_chunks)
    ]

    new_faiss.add(vectors, np.arange(len(active_chunks), dtype=np.int64))
    new_meta.add_chunks(list(zip(range(len(active_chunks)), rebuilt_chunks, strict=False)))
    new_faiss.save(str(new_ver_dir / "index.faiss"))

    # Write manifest and activate
    index_mgr.version_manager.write_manifest_and_checksum(
        new_ver_dir,
        dimension=index_mgr.dimension,
        index_type=index_mgr.index_type,
        documents_count=len(set(c["document_id"] for c in active_chunks)),
        vectors_count=len(active_chunks),
    )
    index_mgr.version_manager.activate_version(new_ver_name, expected_dimension=index_mgr.dimension)
    index_mgr.load_active_index()
    engine.sync_lexical_index()

    return {
        "status": "rebuilt",
        "new_version": new_ver_name,
        "compacted_vectors": len(active_chunks),
    }
