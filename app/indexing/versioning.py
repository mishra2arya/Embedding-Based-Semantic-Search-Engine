"""Index versioning, validation, atomic activation, and rollback management."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path

import faiss

from app.core.exceptions import IndexBuildError

logger = logging.getLogger(__name__)


class IndexVersionManager:
    """Manages immutable index versions with atomic symlink/pointer activation and rollback."""

    def __init__(self, base_index_dir: Path | str = "./data/indexes"):
        self.base_dir = Path(base_index_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_file = self.base_dir / "CURRENT_VERSION"

    def get_next_version_name(self) -> str:
        """Compute next version name e.g. v001, v002."""
        existing = self.list_version_names()
        if not existing:
            return "v001"
        latest_num = 0
        for v in existing:
            if v.startswith("v") and v[1:].isdigit():
                latest_num = max(latest_num, int(v[1:]))
        return f"v{latest_num + 1:03d}"

    def list_version_names(self) -> list[str]:
        """List all version directories sorted."""
        versions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("v") and item.name[1:].isdigit():
                versions.append(item.name)
        return sorted(versions)

    def get_version_dir(self, version_name: str) -> Path:
        return self.base_dir / version_name

    def compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def write_manifest_and_checksum(
        self,
        version_dir: Path,
        dimension: int = 512,
        index_type: str = "HNSW",
        metric: str = "cosine",
        documents_count: int = 0,
        vectors_count: int = 0,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        extra: dict | None = None,
    ) -> dict:
        """Generate manifest.json and checksum.sha256 in the version directory."""
        index_file = version_dir / "index.faiss"
        meta_file = version_dir / "metadata.db"

        index_checksum = self.compute_file_checksum(index_file) if index_file.exists() else ""
        meta_checksum = self.compute_file_checksum(meta_file) if meta_file.exists() else ""

        manifest = {
            "index_version": version_dir.name,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "dimension": dimension,
            "index_type": index_type,
            "metric": metric,
            "documents": documents_count,
            "vectors": vectors_count,
            "model": model_name,
            "checksums": {
                "index.faiss": index_checksum,
                "metadata.db": meta_checksum,
            },
            "extra": extra or {},
        }

        manifest_path = version_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Write aggregate checksum file
        agg_hasher = hashlib.sha256()
        agg_hasher.update((index_checksum + meta_checksum).encode("utf-8"))
        checksum_val = agg_hasher.hexdigest()

        with open(version_dir / "checksum.sha256", "w", encoding="utf-8") as f:
            f.write(checksum_val)

        manifest["checksum"] = checksum_val
        return manifest

    def validate_version(self, version_dir: Path, expected_dimension: int = 512) -> bool:
        """Validate that a version is complete, uncorrupted, and queryable."""
        if not version_dir.exists() or not version_dir.is_dir():
            logger.error(f"Version directory does not exist: {version_dir}")
            return False

        index_file = version_dir / "index.faiss"
        meta_file = version_dir / "metadata.db"
        manifest_file = version_dir / "manifest.json"

        if not (index_file.exists() and meta_file.exists() and manifest_file.exists()):
            logger.error(f"Missing essential files in {version_dir}")
            return False

        try:
            # Check manifest
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)

            if manifest.get("dimension") != expected_dimension:
                logger.error(
                    f"Manifest dimension mismatch: {manifest.get('dimension')} != {expected_dimension}"
                )
                return False

            # Verify FAISS load and dimension
            index = faiss.read_index(str(index_file))
            if index.d != expected_dimension:
                logger.error(f"FAISS index dimension mismatch: {index.d} != {expected_dimension}")
                return False

            # Check search viability if vectors exist
            if index.ntotal > 0:
                dummy_q = faiss.randn((1, expected_dimension), 1234)
                faiss.normalize_L2(dummy_q)
                index.search(dummy_q, min(5, index.ntotal))

            logger.info(
                f"Version {version_dir.name} validated successfully ({index.ntotal} vectors)."
            )
            return True
        except Exception as e:
            logger.error(f"Validation failed for {version_dir.name}: {e}")
            return False

    def activate_version(self, version_name: str, expected_dimension: int = 512) -> None:
        """Validate and atomically activate a version."""
        version_dir = self.get_version_dir(version_name)
        if not self.validate_version(version_dir, expected_dimension=expected_dimension):
            raise IndexBuildError(f"Cannot activate invalid index version: {version_name}")

        current_symlink = self.base_dir / "current"
        # Atomically update symlink or write pointer file
        try:
            if current_symlink.is_symlink() or current_symlink.exists():
                current_symlink.unlink()
            current_symlink.symlink_to(version_dir.resolve(), target_is_directory=True)
        except Exception as e:
            logger.warning(
                f"Could not create symlink ({e}), falling back to CURRENT_VERSION pointer."
            )

        # Always write pointer file for portable reading across all platforms
        with open(self.pointer_file, "w", encoding="utf-8") as f:
            f.write(version_name.strip())

        logger.info(f"Successfully activated index version '{version_name}'.")

    def get_active_version_name(self) -> str | None:
        """Return name of currently active version."""
        if self.pointer_file.exists():
            with open(self.pointer_file, encoding="utf-8") as f:
                v = f.read().strip()
                if v:
                    return v

        current_symlink = self.base_dir / "current"
        if current_symlink.is_symlink():
            target = current_symlink.resolve()
            return target.name

        return None

    def get_active_version_dir(self) -> Path | None:
        active_name = self.get_active_version_name()
        if active_name:
            v_dir = self.get_version_dir(active_name)
            if v_dir.exists():
                return v_dir
        return None

    def rollback_version(self, expected_dimension: int = 512) -> str:
        """Rollback to previous valid version."""
        current = self.get_active_version_name()
        versions = self.list_version_names()

        if not current or current not in versions:
            raise IndexBuildError("No active version found to rollback from.")

        curr_idx = versions.index(current)
        if curr_idx == 0:
            raise IndexBuildError(
                f"Current version '{current}' is the oldest version. Cannot rollback."
            )

        prev_version = versions[curr_idx - 1]
        self.activate_version(prev_version, expected_dimension=expected_dimension)
        return prev_version
