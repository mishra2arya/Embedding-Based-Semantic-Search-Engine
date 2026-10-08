# Security, RBAC & Multi-Tenant Isolation

## 1. Authentication & RBAC
Authentication is enforced via the `X-API-Key` HTTP header. Three distinct roles are supported:

| Role | Allowed Endpoints | Description |
| :--- | :--- | :--- |
| `readonly` | `/api/v1/search`, `/api/v1/rag`, `/api/v1/documents` (GET), `/api/v1/health` | Read-only search and inspection |
| `operator` | Readonly permissions + `/api/v1/ingest`, `/api/v1/documents/{id}` (DELETE) | Data management and ingestion |
| `admin` | Operator permissions + `/api/v1/index/*` (rebuild, activate, rollback) | Full cluster and index control |

## 2. Multi-Tenant Isolation
Every document and chunk maintains a `tenant_id` attribute.
- Ingestion enforces tenant metadata tagging.
- Retrieval queries strictly filter candidate vectors matching the caller's `tenant_id`.
- Queries originating from `tenant_a` will never retrieve chunks belonging to `tenant_b`.

## 3. Defense Against Malicious Payloads
- **Path Traversal Defense**: Filenames are strictly sanitized to prevent `../` directory escapes.
- **Payload Limits**: Max file size defaults to 25 MB; query length is bounded at 2,000 characters.
- **Rate Limiting**: Sliding-window rate limiting throttles abusive clients (default 120 req/min).
- **Prompt Injection Neutralization**: Untrusted retrieved content is isolated within XML fences and stripped of prompt override instructions.
