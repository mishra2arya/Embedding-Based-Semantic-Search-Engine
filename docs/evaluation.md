# Evaluation Framework & Information Retrieval Metrics

## Evaluation Methodology
The evaluation suite benchmarks retrieval quality across 500 ground-truth queries categorized into 8 distinct challenge sets:
1. **Factual**: Direct entity and port/version lookups.
2. **Technical**: Architectural explanations and algorithmic nuances.
3. **Ambiguous**: Broad terms spanning multiple contexts.
4. **Multi-Hop**: Queries requiring synthesized reasoning across concepts.
5. **Long-Form**: Comprehensive paragraph-length queries.
6. **Keyword-Heavy**: Dense technical acronyms and cipher names.
7. **Semantic**: Conceptual paraphrases without exact lexical overlap.
8. **Adversarial**: Prompt injection and extraction probes.

## Evaluated Metrics
- **Precision@10**: Proportion of retrieved chunks in top-10 that are relevant.
- **Recall@10**: Proportion of all relevant chunks captured in top-10.
- **MRR@10 (Mean Reciprocal Rank)**: Quality of top-ranked hit position ($1 / \text{rank}$).
- **nDCG@10**: Position-weighted cumulative gain with logarithmic decay.
- **Hit Rate@10**: Percentage of queries with at least one hit in top-10.

## Running Evaluation
```bash
python scripts/evaluate.py --index data/indexes --top-k 10
# or using Make
make evaluate
```
Reports are automatically written to `benchmarks/benchmark.md`, `benchmarks/benchmark.csv`, and `benchmarks/benchmark.json`.
