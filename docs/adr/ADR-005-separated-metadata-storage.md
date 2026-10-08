# ADR-005: Separation of Vector Index from Metadata Storage

## Context & Problem Statement
FAISS is an optimized numeric vector mathematics library. It operates strictly on integer IDs and float arrays; it does not support arbitrary text storage, rich JSON metadata, or relational filter predicates (such as tenant isolation, date ranges, tags, or author matching).

## Decision
We implemented a **Separated Metadata Architecture** pairing FAISS with an embedded **SQLite metadata database** running in WAL (Write-Ahead Logging) mode.

## Architecture
```text
FAISS Vector ID (int64)
       │
       ▼
SQLite `chunks` table (chunk_id, document_id, text, metadata_json, is_tombstone)
       │
       ▼
SQLite `documents` table (document_id, title, source, tenant_id, metadata_json)
```

## Trade-Offs & Rationale
1. **Separation of Concerns**: Allows FAISS to execute pure SIMD-vector arithmetic in C++ without bloat from string objects or heap fragmentation.
2. **ACID Transactions & Filtering**: SQLite provides rich SQL indexing, JSON extraction (`json_extract`), and ACID transactions for document deletions and updates.
3. **Zero External Daemon**: SQLite is zero-configuration, embeds directly into the Python process, and writes to `metadata.db` alongside `index.faiss`.
4. **Instant Tombstoning**: Document deletion marks `is_tombstone = 1`, instantaneously excluding deleted documents from search without requiring an expensive full FAISS index rebuild.

## Consequences
- Requires two-stage lookup during retrieval: candidate IDs from FAISS are looked up in SQLite to hydrate text and apply metadata filters. Because lookups use primary-key B-tree indexes, latency is < 1 millisecond.
