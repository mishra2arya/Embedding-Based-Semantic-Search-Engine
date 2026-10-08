"""Unit tests for SQLite metadata storage, CRUD, and tombstones."""

from pathlib import Path

from app.indexing.persistence import MetadataStore
from app.ingestion.metadata import ChunkRecord, DocumentRecord


def test_metadata_store_crud(tmp_path: Path):
    db_file = tmp_path / "metadata.db"
    store = MetadataStore(db_file)

    doc = DocumentRecord(
        document_id="doc_test_1",
        title="Test Document",
        source="test",
        checksum="hash1",
        tenant_id="default",
        metadata={"category": "security", "author": "Alice"},
    )
    chunk = ChunkRecord(
        chunk_id="chk_test_1_0",
        document_id="doc_test_1",
        chunk_index=0,
        text="Sample chunk content text.",
        checksum="chash1",
        tenant_id="default",
    )

    store.add_documents([doc])
    store.add_chunks([(0, chunk)])

    assert store.count_documents() == 1
    assert store.count_chunks() == 1

    # Retrieve chunk
    retrieved = store.get_chunks_by_faiss_ids([0])
    assert 0 in retrieved
    assert retrieved[0]["title"] == "Test Document"
    assert retrieved[0]["metadata"]["author"] == "Alice"

    # Test filtering
    matching_ids = store.get_matching_faiss_ids(
        tenant_id="default", filters={"category": "security"}
    )
    assert 0 in matching_ids

    non_matching = store.get_matching_faiss_ids(
        tenant_id="default", filters={"category": "finance"}
    )
    assert 0 not in non_matching

    # Test tombstone deletion
    success = store.delete_document("doc_test_1", tenant_id="default")
    assert success
    assert store.count_documents() == 0
    # Chunk should now be tombstoned
    after_del = store.get_chunks_by_faiss_ids([0])
    assert 0 not in after_del
