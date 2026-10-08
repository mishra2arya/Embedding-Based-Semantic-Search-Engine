"""API endpoint tests validating request/response schemas, status codes, and headers."""

import uuid

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_health_and_readiness_endpoints():
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_ready = client.get("/api/v1/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["ready"] is True


def test_metrics_endpoint():
    r_metrics = client.get("/api/v1/metrics")
    assert r_metrics.status_code == 200
    assert "search_requests_total" in r_metrics.text


def test_ingest_and_search_endpoints():
    unique_id = uuid.uuid4().hex[:8]
    tenant = f"tenant_{unique_id}"
    title = f"API Testing Document {unique_id}"
    text = f"FastAPI provides high performance ASGI web routing with Pydantic validation. Code: {unique_id}"

    # Ingest document
    r_ingest = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": settings.api_key_admin},
        json={
            "title": title,
            "text": text,
            "tenant_id": tenant,
        },
    )
    assert r_ingest.status_code == 200
    doc_id = r_ingest.json()["document_id"]
    assert doc_id.startswith("doc_")

    # Search document
    r_search = client.post(
        "/api/v1/search",
        headers={"X-API-Key": settings.api_key_readonly},
        json={
            "query": f"FastAPI ASGI routing {unique_id}",
            "tenant_id": tenant,
            "top_k": 3,
        },
    )
    assert r_search.status_code == 200
    results = r_search.json()["results"]
    assert len(results) > 0
    assert results[0]["document_id"] == doc_id
    assert "citation" in results[0]

    # RAG endpoint
    r_rag = client.post(
        "/api/v1/rag",
        headers={"X-API-Key": settings.api_key_readonly},
        json={"query": "What does FastAPI provide?", "tenant_id": tenant},
    )
    assert r_rag.status_code == 200
    assert len(r_rag.json()["sources"]) > 0


def test_document_management_endpoints():
    unique_id = uuid.uuid4().hex[:8]
    tenant = f"tenant_mgmt_{unique_id}"

    # Ingest document first to ensure isolation
    r_ingest = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": settings.api_key_admin},
        json={
            "title": f"Doc Mgmt Test {unique_id}",
            "text": f"Document for management testing. Unique ID {unique_id}",
            "tenant_id": tenant,
        },
    )
    assert r_ingest.status_code == 200
    doc_id = r_ingest.json()["document_id"]

    r_list = client.get(
        f"/api/v1/documents?tenant_id={tenant}",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_list.status_code == 200
    docs = r_list.json()["documents"]
    assert len(docs) > 0
    assert any(d["document_id"] == doc_id for d in docs)

    # Delete document
    r_del = client.delete(
        f"/api/v1/documents/{doc_id}?tenant_id={tenant}",
        headers={"X-API-Key": settings.api_key_operator},
    )
    assert r_del.status_code == 200



def test_index_status_endpoint():
    r_status = client.get(
        "/api/v1/index/status",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_status.status_code == 200
    data = r_status.json()
    assert data["dimension"] == 512
    assert "active_version" in data


def test_analytics_telemetry_and_collections():
    r_tel = client.get(
        "/api/v1/analytics/telemetry",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_tel.status_code == 200
    tel_data = r_tel.json()
    assert tel_data["status"] == "healthy"
    assert "cpu_percent" in tel_data
    assert "memory_rss_mb" in tel_data
    assert tel_data["dimension"] == 512

    r_col = client.get(
        "/api/v1/analytics/collections",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_col.status_code == 200
    collections = r_col.json()
    assert isinstance(collections, list)


def test_ingest_file_endpoint():
    unique_id = uuid.uuid4().hex[:8]
    tenant = f"tenant_file_{unique_id}"
    file_bytes = f"Uploading text file content for indexing test {unique_id}".encode()

    r_file = client.post(
        "/api/v1/ingest/file",
        headers={"X-API-Key": settings.api_key_operator},
        files={"file": ("upload.txt", file_bytes, "text/plain")},
        data={"tenant_id": tenant, "title": f"Upload Doc {unique_id}"},
    )
    assert r_file.status_code == 200
    res = r_file.json()
    assert res["status"] == "indexed"
    assert res["chunks_created"] > 0
    assert res["tenant_id"] == tenant

    # Ingest duplicate file
    r_dup = client.post(
        "/api/v1/ingest/file",
        headers={"X-API-Key": settings.api_key_operator},
        files={"file": ("upload.txt", file_bytes, "text/plain")},
        data={"tenant_id": tenant, "title": f"Upload Doc {unique_id}"},
    )
    assert r_dup.status_code == 200
    assert r_dup.json()["status"] == "duplicate_skipped"


def test_analytics_benchmarks_and_activity():
    r_bench = client.get(
        "/api/v1/analytics/benchmarks",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_bench.status_code == 200
    bench_data = r_bench.json()
    assert "retrieval_evaluation" in bench_data
    assert "load_testing" in bench_data

    r_act = client.get(
        "/api/v1/analytics/activity?limit=10",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_act.status_code == 200
    assert isinstance(r_act.json(), list)


def test_analytics_vector_space_and_projection():
    r_vs = client.get(
        "/api/v1/analytics/vector-space?limit=10",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_vs.status_code == 200
    vs_data = r_vs.json()
    assert "points" in vs_data
    assert "clusters" in vs_data

    r_proj = client.post(
        "/api/v1/analytics/vector-space/project",
        headers={"X-API-Key": settings.api_key_readonly},
        json={"query": "kubernetes cluster networking"},
    )
    assert r_proj.status_code == 200
    proj_data = r_proj.json()
    assert "x" in proj_data
    assert "y" in proj_data
    assert proj_data["query"] == "kubernetes cluster networking"


def test_admin_activate_and_rollback_endpoints():
    from unittest.mock import MagicMock

    from app.api.dependencies import get_index_manager, get_search_engine

    mock_mgr = MagicMock()
    mock_mgr.dimension = 512
    mock_mgr.version_manager.rollback_version.return_value = "v001"
    mock_engine = MagicMock()

    app.dependency_overrides[get_index_manager] = lambda: mock_mgr
    app.dependency_overrides[get_search_engine] = lambda: mock_engine

    try:
        # Activate
        r_act = client.post(
            "/api/v1/index/activate",
            headers={"X-API-Key": settings.api_key_admin},
            json={"version_name": "v001"},
        )
        assert r_act.status_code == 200
        assert r_act.json()["status"] == "activated"
        mock_mgr.version_manager.activate_version.assert_called_once_with("v001", expected_dimension=512)

        # Rollback
        r_rb = client.post(
            "/api/v1/index/rollback",
            headers={"X-API-Key": settings.api_key_admin},
        )
        assert r_rb.status_code == 200
        assert r_rb.json()["status"] == "rolled_back"
        assert r_rb.json()["active_version"] == "v001"

        # Rebuild when not ready
        mock_mgr.is_ready.return_value = False
        r_reb_err = client.post(
            "/api/v1/index/rebuild",
            headers={"X-API-Key": settings.api_key_admin},
        )
        assert r_reb_err.status_code == 200
        assert r_reb_err.json()["status"] == "error"

        # Rebuild when active chunks empty
        mock_mgr.is_ready.return_value = True
        mock_mgr.metadata_store.get_all_active_chunks.return_value = []
        r_reb_empty = client.post(
            "/api/v1/index/rebuild",
            headers={"X-API-Key": settings.api_key_admin},
        )
        assert r_reb_empty.status_code == 200
        assert r_reb_empty.json()["status"] == "empty"
    finally:
        app.dependency_overrides.pop(get_index_manager, None)
        app.dependency_overrides.pop(get_search_engine, None)


