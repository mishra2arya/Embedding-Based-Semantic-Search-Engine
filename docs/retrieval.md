# Retrieval & Ranking Engine

## Hybrid Search Architecture
The search engine executes dense vector search in FAISS simultaneously with lexical search in Okapi BM25, combining candidate pools using configurable fusion methods.

### 1. Dense Semantic Retrieval
- **Algorithm**: HNSW (Hierarchical Navigable Small World) with cosine distance.
- **Vectors**: 512-dimensional L2-normalized float32 vectors.
- **Latency**: Typically 1–3 ms on 500,000 vectors.

### 2. Lexical Retrieval
- **Algorithm**: Okapi BM25 ($k_1=1.5, b=0.75$) with inverse document frequency normalization.
- **Baseline**: Scikit-Learn TF-IDF for regression benchmarking.

### 3. Candidate Fusion Methods
#### Weighted Linear Combination
Scores are min-max normalized:
$$\text{Score} = w_{\text{sem}} \cdot \text{Norm}(\text{Score}_{\text{sem}}) + w_{\text{lex}} \cdot \text{Norm}(\text{Score}_{\text{lex}})$$
Default weights: $w_{\text{sem}} = 0.7$, $w_{\text{lex}} = 0.3$.

#### Reciprocal Rank Fusion (RRF)
Position-invariant rank fusion:
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{lexical}\}} \frac{1}{k + r_m(d)}$$
Default smoothing constant: $k = 60$.

### 4. Explainable Score Diagnostics
Every search result item returns complete transparency into its scoring journey:
```json
{
  "rank": 1,
  "score": 0.892,
  "semantic_score": 0.845,
  "lexical_score": 12.38,
  "fusion_score": 0.852,
  "rerank_score": 0.892,
  "final_score": 0.892,
  "citation": "[1] \"Title\" (Doc: doc_8c92, Chunk: chk_8c92_0)"
}
```
