"""Main FastAPI Application with structured JSON logging, security, and exception handling."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import app_state
from app.api.routes import admin, analytics, documents, health, ingest, rag, search
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.core.security import rate_limiter
from app.observability.metrics import API_ERRORS_TOTAL

# Setup structured logging
setup_logging(level=settings.log_level, log_format=settings.log_format)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing model and index warmup."""
    logger.info(f"Starting {settings.app_name} in {settings.app_env} environment...")
    app_state.initialize()
    yield
    logger.info("Shutting down application services...")


app = FastAPI(
    title="Production-Grade Semantic Search & RAG Platform",
    description=(
        "Enterprise-scale semantic retrieval and RAG engine capable of indexing, filtering, "
        "and serving 500,000+ documents with FAISS vector indexing, hybrid BM25 fusion, and prompt injection defense."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_lifecycle_middleware(request: Request, call_next):
    """Assign X-Request-ID, apply rate limiting, and output structured JSON access logs."""
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id

    # Exclude static assets and health check from rate limiting
    path = request.url.path
    if not (
        path.startswith("/dashboard")
        or path.startswith("/static")
        or path in ("/api/v1/health", "/api/v1/ready", "/api/v1/metrics", "/")
    ):
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", client_ip)
        try:
            rate_limiter.check(api_key)
        except AppException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )

    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id

        # Structured access logging
        if not path.startswith("/api/v1/metrics"):
            extra = {
                "event": "http_request",
                "request_id": request_id,
                "endpoint": path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": latency_ms,
            }
            logger.info(
                f"{request.method} {path} -> {response.status_code} ({latency_ms}ms)", extra=extra
            )

        return response
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.error(
            f"Unhandled exception on {request.method} {path}: {exc}",
            extra={
                "event": "http_error",
                "request_id": request_id,
                "endpoint": path,
                "latency_ms": latency_ms,
            },
            exc_info=True,
        )
        API_ERRORS_TOTAL.labels(error_code="INTERNAL_SERVER_ERROR", endpoint=path).inc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing the request.",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id},
        )


# Standard error handlers matching Section 55
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    req_id = getattr(request.state, "request_id", "unknown")
    API_ERRORS_TOTAL.labels(error_code=exc.code, endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": req_id,
                "details": exc.details if exc.details else None,
            }
        },
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "unknown")
    API_ERRORS_TOTAL.labels(error_code="VALIDATION_ERROR", endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request payload schema.",
                "request_id": req_id,
                "details": {"errors": exc.errors()},
            }
        },
        headers={"X-Request-ID": req_id},
    )


# Register API v1 routes
app.include_router(search.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

# Dashboard static files mount
dashboard_dir = Path("./dashboard")
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

    @app.get("/", summary="Web Dashboard UI", include_in_schema=False)
    def root():
        index_file = dashboard_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "Semantic Search Engine running. Visit /docs for API documentation."}

    @app.get("/dashboard", summary="Web Dashboard UI", include_in_schema=False)
    def dashboard():
        index_file = dashboard_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "Dashboard files not found."}
