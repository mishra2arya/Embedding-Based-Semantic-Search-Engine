# ADR-002: Adoption of Sentence-Transformers with 512-Dimensional Normalization

## Context & Problem Statement
Semantic search requires converting natural language queries and unstructured text chunks into dense continuous vector representations that capture semantic meaning, synonyms, and conceptual intent.

## Decision
We adopted **Sentence-Transformers** (built on PyTorch) configured with 512-dimensional normalized embeddings and an orthogonal projection layer.

## Trade-Offs & Rationale
1. **Siamese Network Calibration**: Unlike raw BERT models that require costly cross-encoders for every pair, bi-encoder Sentence-Transformers map independent texts into a metric space where cosine distance correlates directly with semantic similarity.
2. **CPU Multi-Core Acceleration**: Utilizing PyTorch with OpenMP/MKL enables parallel batch inference across CPU cores, achieving high throughput without requiring expensive GPU infrastructure.
3. **Unit L2 Normalization**: By enforcing $\|v\|_2 = 1$, the inner product $\langle u, v \rangle$ is mathematically identical to cosine similarity. This permits using fast `IndexFlatIP` and `IndexHNSWFlat` in FAISS.
4. **Dimension Verification & Projection**: Startup validation verifies the 512-dimension guarantee, applying a deterministic orthogonal projection matrix if base models output different dimensions.

## Consequences
- Model weights require initial local caching (~90MB).
- Offline environments use local caches or fast deterministic fallback representations.
