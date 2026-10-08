"""Unit tests for metadata filtering and multi-tenant security boundaries."""

from __future__ import annotations

from app.retrieval.filtering import MetadataFilter


def test_metadata_filter_tenant_isolation():
    chunk = {
        "chunk_id": "c1",
        "tenant_id": "tenant_a",
        "text": "Tenant A isolated data",
        "metadata": {},
    }

    # Match tenant
    assert MetadataFilter.match(chunk, filters=None, tenant_id="tenant_a") is True
    # Different tenant -> rejected
    assert MetadataFilter.match(chunk, filters=None, tenant_id="tenant_b") is False


def test_metadata_filter_predicates():
    chunk = {
        "chunk_id": "c2",
        "tenant_id": "default",
        "source": "api_upload",
        "language": "en",
        "mime_type": "application/pdf",
        "created_at": "2026-06-01T10:00:00",
        "metadata": {
            "category": "technical",
            "author": "Alice",
            "tags": ["cloud", "security", "k8s"],
            "custom_attr": "custom_val",
        },
    }

    # No filters matches everything for tenant
    assert MetadataFilter.match(chunk, filters={}, tenant_id="default") is True

    # Source matching
    assert MetadataFilter.match(chunk, {"source": "api_upload"}) is True
    assert MetadataFilter.match(chunk, {"source": "filesystem"}) is False

    # Language matching
    assert MetadataFilter.match(chunk, {"language": "en"}) is True
    assert MetadataFilter.match(chunk, {"language": "fr"}) is False

    # Mime type matching
    assert MetadataFilter.match(chunk, {"mime_type": "application/pdf"}) is True
    assert MetadataFilter.match(chunk, {"mime_type": "text/plain"}) is False

    # Category matching
    assert MetadataFilter.match(chunk, {"category": "technical"}) is True
    assert MetadataFilter.match(chunk, {"category": "finance"}) is False

    # Author matching
    assert MetadataFilter.match(chunk, {"author": "Alice"}) is True
    assert MetadataFilter.match(chunk, {"author": "Bob"}) is False

    # Created after matching
    assert MetadataFilter.match(chunk, {"created_after": "2026-01-01"}) is True
    assert MetadataFilter.match(chunk, {"created_after": "2026-07-01"}) is False

    # Tags matching: list
    assert MetadataFilter.match(chunk, {"tags": ["k8s", "docker"]}) is True
    assert MetadataFilter.match(chunk, {"tags": ["nonexistent", "other"]}) is False

    # Tags matching: single value
    assert MetadataFilter.match(chunk, {"tags": "security"}) is True
    assert MetadataFilter.match(chunk, {"tags": "missing_tag"}) is False

    # Custom attribute matching
    assert MetadataFilter.match(chunk, {"custom_attr": "custom_val"}) is True
    assert MetadataFilter.match(chunk, {"custom_attr": "wrong_val"}) is False


def test_metadata_filter_candidates():
    candidates = [
        {"chunk_id": "c1", "tenant_id": "t1", "source": "web", "metadata": {}},
        {"chunk_id": "c2", "tenant_id": "t1", "source": "api", "metadata": {}},
        {"chunk_id": "c3", "tenant_id": "t2", "source": "api", "metadata": {}},
    ]

    filtered = MetadataFilter.filter_candidates(candidates, filters={"source": "api"}, tenant_id="t1")
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "c2"
