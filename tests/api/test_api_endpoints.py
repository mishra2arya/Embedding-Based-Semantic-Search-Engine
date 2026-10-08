"""API endpoint tests validating request/response schemas, status codes, and headers."""

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
    # Ingest document
    r_ingest = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": settings.api_key_admin},
        json={
            "title": "API Testing Document",
            "text": "FastAPI provides high performance ASGI web routing with Pydantic validation.",
            "tenant_id": "api_test_tenant",
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
            "query": "FastAPI ASGI routing",
            "tenant_id": "api_test_tenant",
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
        json={"query": "What does FastAPI provide?", "tenant_id": "api_test_tenant"},
    )
    assert r_rag.status_code == 200
    assert len(r_rag.json()["sources"]) > 0


def test_document_management_endpoints():
    r_list = client.get(
        "/api/v1/documents?tenant_id=api_test_tenant",
        headers={"X-API-Key": settings.api_key_readonly},
    )
    assert r_list.status_code == 200
    docs = r_list.json()["documents"]
    assert len(docs) > 0
    doc_id = docs[0]["document_id"]

    # Delete document
    r_del = client.delete(
        f"/api/v1/documents/{doc_id}?tenant_id=api_test_tenant",
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
