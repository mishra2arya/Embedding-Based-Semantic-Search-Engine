# ADR-004: Vector Index Selection: HNSW vs IVF

## Context & Problem Statement
Indexing 500,000+ high-dimensional vectors requires balancing query latency, recall quality, index construction time, and memory consumption.

## Decision
We implemented configurable support for **IndexHNSWFlat**, **IndexIVFFlat**, and **IndexFlatIP**, with **HNSW (Hierarchical Navigable Small World)** as the default production configuration ($M=32, efConstruction=200, efSearch=64$).

## Trade-Offs & Rationale
1. **HNSW vs IVF**:
   - **HNSW**:
     - *Pros*: Superior recall (typically >95% at top-10) with sub-millisecond search latencies on 500k vectors. No training phase required; vectors can be incrementally inserted without clustering re-training.
     - *Cons*: Higher index memory overhead (~1.5x to 2x raw vector size due to graph edge lists).
   - **IVF**:
     - *Pros*: Lower memory footprint; good for massive scales (>10M vectors) when combined with product quantization (IVF-PQ).
     - *Cons*: Requires an initial training phase on representative vectors; recall drops unless $nprobe$ is high, which increases query latency.
2. **500k Scale Compatibility**: At 500,000 512-D vectors, HNSW consumes ~1.8 GB RAM, which easily fits within our 16 GB hardware budget while providing single-digit millisecond retrieval.

## Consequences
- HNSW construction takes slightly longer during initial batch building, but yields optimal runtime query latency.
