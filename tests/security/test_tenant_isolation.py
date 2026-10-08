"""Security tests verifying strict multi-tenant data isolation."""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_strict_tenant_isolation():
    # Ingest document for Tenant Red
    client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": settings.api_key_admin},
        json={
            "title": "Confidential Strategy Red",
            "text": "Top secret corporate financial records for project Redstone.",
            "tenant_id": "tenant_red",
        },
    )

    # Ingest document for Tenant Blue
    client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": settings.api_key_admin},
        json={
            "title": "Confidential Strategy Blue",
            "text": "Top secret intellectual property patents for project Bluefire.",
            "tenant_id": "tenant_blue",
        },
    )

    # Tenant Blue searches for Redstone
    resp_blue = client.post(
        "/api/v1/search",
        headers={"X-API-Key": settings.api_key_readonly},
        json={
            "query": "financial records project Redstone",
            "tenant_id": "tenant_blue",
            "top_k": 10,
        },
    )
    assert resp_blue.status_code == 200
    results_blue = resp_blue.json()["results"]
    for r in results_blue:
        assert "Redstone" not in r["text"]
        assert r["metadata"].get("tenant_id", "tenant_blue") == "tenant_blue"

    # Tenant Red searches for Redstone
    resp_red = client.post(
        "/api/v1/search",
        headers={"X-API-Key": settings.api_key_readonly},
        json={
            "query": "financial records project Redstone",
            "tenant_id": "tenant_red",
            "top_k": 5,
        },
    )
    assert resp_red.status_code == 200
    assert any("Redstone" in r["text"] for r in resp_red.json()["results"])
