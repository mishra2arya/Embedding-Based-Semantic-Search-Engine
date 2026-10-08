# API Reference & Curl Examples

Interactive OpenAPI Swagger UI is available at `http://localhost:8000/docs`.

## Authentication Header
All authenticated endpoints require:
```text
X-API-Key: <your-api-key>
```

---

## 1. Search API
**Endpoint:** `POST /api/v1/search`

### Curl Example:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: admin-secret-key-12345" \
  -d '{
    "query": "How does TLS protect network communication?",
    "top_k": 5,
    "mode": "hybrid",
    "rerank": true,
    "filters": {
      "category": "cybersecurity"
    }
  }'
```

---

## 2. RAG API
**Endpoint:** `POST /api/v1/rag`

### Curl Example:
```bash
curl -X POST http://localhost:8000/api/v1/rag \
  -H "Content-Type: application/json" \
  -H "X-API-Key: readonly-secret-key-12345" \
  -d '{
    "query": "What is Zero Trust Network Access and how does it work?",
    "top_k": 3,
    "rerank": true
  }'
```

---

## 3. Ingestion API
**Endpoint:** `POST /api/v1/ingest`

### Text Ingestion Example:
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: operator-secret-key-12345" \
  -d '{
    "title": "BGP Routing Architecture",
    "text": "Border Gateway Protocol manages packet routing across autonomous systems.",
    "tenant_id": "default",
    "metadata": {"category": "networking"}
  }'
```

### File Upload Example:
```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -H "X-API-Key: operator-secret-key-12345" \
  -F "file=@document.pdf" \
  -F "tenant_id=default"
```

---

## 4. Document Management
**List Documents:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents?limit=20&tenant_id=default" \
  -H "X-API-Key: readonly-secret-key-12345"
```

**Delete Document:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/doc_8c92?tenant_id=default" \
  -H "X-API-Key: operator-secret-key-12345"
```

---

## 5. Index Administration
**Get Index Status:**
```bash
curl -X GET http://localhost:8000/api/v1/index/status \
  -H "X-API-Key: admin-secret-key-12345"
```

**Rebuild & Compact Index:**
```bash
curl -X POST http://localhost:8000/api/v1/index/rebuild \
  -H "X-API-Key: admin-secret-key-12345"
```

---

## 6. Health & Metrics
**Liveness:**
```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Readiness:**
```bash
curl -X GET http://localhost:8000/api/v1/ready
```

**Prometheus Metrics:**
```bash
curl -X GET http://localhost:8000/api/v1/metrics
```
