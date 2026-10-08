# ADR-003: Hybrid Retrieval Combining Dense FAISS and Okapi BM25

## Context & Problem Statement
Pure dense retrieval struggles with exact keyword queries, rare identifiers (e.g. error codes, UUIDs, model numbers, RFC references), and out-of-vocabulary terms. Conversely, pure lexical search (TF-IDF/BM25) fails on paraphrases, conceptual queries, and vocabulary mismatch.

## Decision
We implemented a **Hybrid Retrieval Architecture** combining dense vector retrieval (FAISS) and lexical retrieval (Okapi BM25) coupled with candidate fusion (Weighted Score Fusion and Reciprocal Rank Fusion).

## Trade-Offs & Rationale
1. **Complementary Strengths**:
   - Dense retrieval provides high semantic recall for paraphrased, descriptive, and conceptual queries.
   - BM25 delivers high precision on exact terms, technical keywords, acronyms, and product IDs.
2. **Robust Candidate Fusion**:
   - **Weighted Score Fusion**: Normalizes scores from disparate distributions to combine strengths via configurable weights (`0.7` semantic, `0.3` lexical).
   - **Reciprocal Rank Fusion (RRF)**: Position-based fusion invariant to raw score magnitudes ($1 / (k + rank)$), ensuring high stability across diverse query types.
3. **Empirical Superiority**: Benchmark experiments consistently demonstrate that Hybrid retrieval achieves higher nDCG@10 and Recall@10 than either pure dense or pure lexical search.

## Consequences
- Requires synchronizing the in-memory/persisted BM25 inverted index alongside the FAISS vector index.
- Slight increase in candidate gathering latency (~0.2ms), which remains well within our <50ms P50 latency budget.
