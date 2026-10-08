"""Health, readiness, and Prometheus metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_health_checker
from app.observability.health import HealthChecker
from app.observability.metrics import get_metrics_output

router = APIRouter(tags=["Health & Observability"])


@router.get("/health", summary="Liveness probe", status_code=status.HTTP_200_OK)
def liveness(checker: HealthChecker = Depends(get_health_checker)):
    """Kubernetes liveness probe: verifies process is alive and responsive."""
    return checker.check_liveness()


@router.get("/ready", summary="Readiness probe")
def readiness(response: Response, checker: HealthChecker = Depends(get_health_checker)):
    """Kubernetes readiness probe: verifies embedding model and FAISS index are loaded."""
    status_data = checker.check_readiness()
    if not status_data["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK
    return status_data


@router.get("/metrics", summary="Prometheus metrics")
def metrics():
    """Prometheus exposition format metrics scraper endpoint."""
    output = get_metrics_output()
    return Response(content=output, media_type="text/plain; version=0.0.4; charset=utf-8")
