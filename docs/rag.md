# Retrieval-Augmented Generation (RAG) & Citations

## End-to-End Pipeline
```text
User Question
     ↓
Query Preprocessing & Defense Checks
     ↓
Hybrid Retrieval & Metadata Filtering
     ↓
Reranker Candidate Ordering
     ↓
Context Optimization (Deduplication, Token Limits, XML Fences)
     ↓
System Prompt Boundary Isolation
     ↓
Provider-Agnostic LLM
     ↓
Attributed Answer with Grounded Citations
```

## Prompt Injection Defense Architecture
Retrieved documents are untrusted data. The platform implements multi-layer defenses:
1. **Clear Delimitation**: Context is enclosed in structured `<document>` blocks with explicit XML index tags.
2. **System Rule Precedence**: The system prompt instructs the generator that untrusted text cannot override safety rules or system instructions.
3. **Adversarial Pattern Filtering**: Common injection signatures (e.g. `ignore all previous instructions`, `<|im_start|>`, `system:`) are detected and neutralized.

## Verified Citation Guarantee
Every answer returned by the `/api/v1/rag` endpoint provides verified source attribution:
```json
{
  "answer": "Transport Layer Security (TLS) encrypts communications using symmetric ciphers. [1]",
  "sources": [
    {
      "document_id": "doc_tls",
      "chunk_id": "chk_tls_0",
      "title": "TLS Architecture",
      "relevance_score": 0.892
    }
  ],
  "retrieval_latency_ms": 12.4,
  "generation_latency_ms": 1.2
}
```
Fabricated citations are prevented by validating reference tokens against the retrieved chunk IDs.
