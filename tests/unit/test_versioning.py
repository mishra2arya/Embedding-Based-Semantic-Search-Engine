"""Unit tests for index versioning, validation, and rollback."""

from pathlib import Path

import faiss

from app.indexing.versioning import IndexVersionManager


def test_index_versioning_lifecycle(tmp_path: Path):
    vm = IndexVersionManager(tmp_path)
    assert vm.get_next_version_name() == "v001"

    # Create v001
    v1_dir = vm.get_version_dir("v001")
    v1_dir.mkdir(parents=True)
    idx1 = faiss.IndexFlatIP(512)
    faiss.write_index(idx1, str(v1_dir / "index.faiss"))
    (v1_dir / "metadata.db").touch()
    vm.write_manifest_and_checksum(v1_dir, dimension=512)

    assert vm.validate_version(v1_dir, expected_dimension=512)
    vm.activate_version("v001", expected_dimension=512)
    assert vm.get_active_version_name() == "v001"

    # Create v002
    assert vm.get_next_version_name() == "v002"
    v2_dir = vm.get_version_dir("v002")
    v2_dir.mkdir(parents=True)
    idx2 = faiss.IndexFlatIP(512)
    faiss.write_index(idx2, str(v2_dir / "index.faiss"))
    (v2_dir / "metadata.db").touch()
    vm.write_manifest_and_checksum(v2_dir, dimension=512)

    vm.activate_version("v002", expected_dimension=512)
    assert vm.get_active_version_name() == "v002"

    # Test rollback to v001
    rolled = vm.rollback_version(expected_dimension=512)
    assert rolled == "v001"
    assert vm.get_active_version_name() == "v001"
