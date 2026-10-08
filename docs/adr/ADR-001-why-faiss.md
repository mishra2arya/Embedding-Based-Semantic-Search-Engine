# ADR-001: Selection of FAISS as Vector Indexing Engine

## Context & Problem Statement
The platform must index and query 500,000+ high-dimensional (512-D) vector embeddings with P95 latency under 80ms while maintaining full operational reproducibility and minimal infrastructure overhead.

## Decision
We selected **FAISS (Facebook AI Similarity Search)** as the core vector search engine.

## Trade-Offs & Rationale
1. **In-Process Performance**: FAISS runs as an in-process C++ library with highly tuned AVX2/AVX-512 vector arithmetic. It incurs zero RPC/HTTP network round-trip overhead compared to distributed vector databases (e.g. Milvus, Pinecone, Qdrant).
2. **Deterministic Index Structures**: Provides direct access to proven index structures (`IndexHNSWFlat`, `IndexIVFFlat`, `IndexFlatIP`).
3. **Memory Footprint & Hardware Efficiency**: Efficient memory management allows 500,000 512-dimensional float32 vectors (~1 GB raw data) to reside comfortably in host RAM.
4. **Offline Reproducibility**: Self-contained file-based serialization (`index.faiss`) enables versioning, atomic symlink pointer swaps, and standalone offline execution without running database daemons.

## Consequences
- FAISS does not natively provide arbitrary string metadata filtering; we designed an external ACID metadata layer (SQLite) with pre/post filtering.
- True point deletions require vector rebuilds or tombstones; we implemented a tombstone mechanism combined with periodic compaction.
