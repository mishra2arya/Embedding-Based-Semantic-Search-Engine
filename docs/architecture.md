# System Architecture & Technical Specifications

## 1. High-Level Architecture Overview

The platform is designed as an enterprise search and RAG retrieval infrastructure capable of indexing and querying 500,000+ documents with sub-100ms latency.

```text
                         ┌────────────────────────────────────────┐
                         │       Web Client / API Consumers       │
                         │   Search / RAG / Ingest / Dashboard    │
                         └───────────────────┬────────────────────┘
                                             │ HTTP REST / JSON
                                             ▼
                         ┌────────────────────────────────────────┐
                         │               FastAPI                  │
                         │  - RBAC Security (Admin/Op/Readonly)   │
                         │  - Sliding-Window Rate Limiting        │
                         │  - Request Profiler & JSON Logging     │
                         │  - Prometheus Metrics Registry         │
                         └───────────────────┬────────────────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
        │  Query Service  │         │ Ingestion Pipe  │         │  Admin Service  │
        │ - Preprocessing │         │ - File Parsers  │         │ - Index Status  │
        │ - Normalization │         │ - Dedup SHA-256 │         │ - Versioning    │
        │ - Security Guard│         │ - 4 Chunk Types │         │ - Rebuild/Rollbk│
        └────────┬────────┘         └────────┬────────┘         └─────────────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Embedding Model │         │ Embedding Model │
        │ - 512-D Norm    │         │ - Batched OpenMP│
        │ - LRU Cache     │         │ - 512-D Orthog  │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │               Retrieval Layer               │
        │                                             │
        │  ┌───────────────┐       ┌───────────────┐  │
        │  │  FAISS HNSW   │       │   Okapi BM25  │  │
        │  │ (Dense Cosine)│       │   (Lexical)   │  │
        │  └───────┬───────┘       └───────┬───────┘  │
        │          │                       │          │
        │          └───────────┬───────────┘          │
        │                      ▼                      │
        │          ┌───────────────────────┐          │
        │          │ Candidate Fusion      │          │
        │          │ (Weighted / RRF)      │          │
        │          └───────────┬───────────┘          │
        │                      ▼                      │
        │          ┌───────────────────────┐          │
        │          │ SQLite Metadata Store │          │
        │          │ (Tenant & Predicates) │          │
        │          └───────────┬───────────┘          │
        │                      ▼                      │
        │          ┌───────────────────────┐          │
        │          │ Reranking (Optional)  │          │
        │          └───────────────────────┘          │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │             RAG Context Builder             │
        │ - Deduplication & Token Budget Optimization │
        │ - Prompt Injection Boundaries & Sanitization│
        │ - Citation Attribution & Provenance Link    │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │            Provider-Agnostic LLM            │
        │      (Local Deterministic / OpenAI)         │
        └─────────────────────────────────────────────┘
```

## 2. Core Subsystems

### Ingestion Subsystem
- **Supported Formats**: TXT, Markdown, HTML, JSON, CSV, PDF, DOCX.
- **Deduplication**: SHA-256 hash of normalized text prevents duplicate documents and chunks.
- **Dynamic Chunking**: 4 strategies (Token sliding window, Sentence boundary, Semantic embedding breakpoint, and Structure aware).

### Embedding Engine
- **Model**: Sentence-Transformers with PyTorch multi-threading.
- **Vector Dimension**: 512 dimensions enforced with deterministic orthogonal projection for mismatched raw models.
- **Normalization**: L2 normalized for cosine similarity via inner product.
- **Cache**: In-memory LRU cache for query vectors.

### Vector & Metadata Storage
- **FAISS**: `IndexHNSWFlat` with $M=32$, $efConstruction=200$, and runtime $efSearch=64$.
- **Metadata**: SQLite with WAL mode, index on `faiss_id`, `tenant_id`, and `is_tombstone`.
- **Versioning**: Atomic symlink activation (`indexes/v001`, `indexes/v002`, `current`).

### Retrieval & Reranking
- **Hybrid Retrieval**: Dense cosine similarity + Okapi BM25 score fusion.
- **Fusion**: Min-max normalized Weighted Score Fusion and Reciprocal Rank Fusion ($k=60$).
- **Reranker**: Two-stage alignment reranker scoring query phrase matches, coverage, and headings.
