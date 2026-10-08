# ADR-006: Selection of FastAPI as API Gateway and Service Framework

## Context & Problem Statement
The retrieval service requires a modern, high-throughput, low-latency REST API layer with strict schema validation, asynchronous request handling, standards-compliant OpenAPI documentation, and robust dependency injection.

## Decision
We selected **FastAPI** running on **Uvicorn** with Starlette.

## Trade-Offs & Rationale
1. **Async IO & Throughput**: Native asynchronous concurrency allows non-blocking handling of concurrent I/O operations (file uploads, health probes, metrics scraping) while delegating CPU-heavy tensor computations to worker threads.
2. **Pydantic Data Validation**: Automatic type validation and serialization catch malformed inputs, oversized queries, and invalid payloads at the gateway boundary before hitting backend search routines.
3. **Automated OpenAPI Documentation**: Generates interactive Swagger (`/docs`) and ReDoc (`/redoc`) specifications automatically from route type signatures without manual documentation drift.
4. **Clean Dependency Injection**: Modular dependency system simplifies injecting configurations, security credentials, index managers, and observability components across routes.

## Consequences
- Requires Python 3.10+ type annotation standards, which aligns with our modern Python 3.13+ target environment.
