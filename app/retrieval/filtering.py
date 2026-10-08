"""Metadata filtering and tenant boundary validation."""

from __future__ import annotations

from typing import Any


class MetadataFilter:
    """Evaluates filter predicates against candidate chunk metadata in memory or queries."""

    @staticmethod
    def match(chunk_data: dict, filters: dict[str, Any] | None, tenant_id: str = "default") -> bool:
        # Enforce strict tenant isolation
        doc_tenant = chunk_data.get("tenant_id", "default")
        if doc_tenant != tenant_id:
            return False

        if not filters:
            return True

        metadata = chunk_data.get("metadata", {})

        for key, expected_val in filters.items():
            if key == "source":
                if chunk_data.get("source") != expected_val:
                    return False
            elif key == "language":
                if chunk_data.get("language") != expected_val:
                    return False
            elif key == "mime_type":
                if chunk_data.get("mime_type") != expected_val:
                    return False
            elif key == "category":
                if metadata.get("category") != expected_val:
                    return False
            elif key == "author":
                if metadata.get("author") != expected_val:
                    return False
            elif key == "created_after":
                created_at = chunk_data.get("created_at") or metadata.get("created_at", "")
                if created_at < str(expected_val):
                    return False
            elif key == "tags":
                tags = metadata.get("tags", [])
                if isinstance(expected_val, list):
                    if not any(t in tags for t in expected_val):
                        return False
                else:
                    if expected_val not in tags:
                        return False
            else:
                # Custom metadata field
                if metadata.get(key) != expected_val:
                    return False

        return True

    @classmethod
    def filter_candidates(
        cls,
        candidates: list[dict],
        filters: dict[str, Any] | None,
        tenant_id: str = "default",
    ) -> list[dict]:
        """Filter a list of candidate chunk dictionaries."""
        return [c for c in candidates if cls.match(c, filters=filters, tenant_id=tenant_id)]
