"""Unit tests for lexical search, candidate fusion, and reranking."""

from app.retrieval.fusion import normalize_scores, reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.lexical import BM25Retriever, TfidfRetriever, simple_tokenize
from app.retrieval.reranking import FastAlignmentReranker


def test_simple_tokenize():
    tokens = simple_tokenize("Zero-Trust Security, mTLS & RFC-7540!")
    assert "zero" in tokens
    assert "trust" in tokens
    assert "security" in tokens
    assert "mtls" in tokens


def test_bm25_and_tfidf():
    corpus = [
        {
            "chunk_id": "c1",
            "text": "TLS encrypts communications using cryptography.",
            "faiss_id": 0,
        },
        {
            "chunk_id": "c2",
            "text": "Kubernetes automates deployment and container scaling.",
            "faiss_id": 1,
        },
    ]

    bm25 = BM25Retriever()
    bm25.fit(corpus)
    res_bm25 = bm25.search("cryptography TLS", top_k=2)
    assert len(res_bm25) > 0
    assert res_bm25[0]["chunk_id"] == "c1"

    tfidf = TfidfRetriever()
    tfidf.fit(corpus)
    res_tfidf = tfidf.search("container deployment", top_k=2)
    assert len(res_tfidf) > 0
    assert res_tfidf[0]["chunk_id"] == "c2"


def test_score_normalization():
    raw = [10.0, 20.0, 30.0]
    norm = normalize_scores(raw)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0


def test_fusion_algorithms():
    sem = [{"chunk_id": "c1", "score": 0.9}, {"chunk_id": "c2", "score": 0.4}]
    lex = [{"chunk_id": "c2", "score": 15.0}, {"chunk_id": "c3", "score": 8.0}]

    weighted = weighted_score_fusion(sem, lex, semantic_weight=0.6, lexical_weight=0.4, top_k=3)
    assert len(weighted) == 3
    assert all("fusion_score" in w for w in weighted)

    rrf = reciprocal_rank_fusion(sem, lex, rrf_k=60, top_k=3)
    assert len(rrf) == 3
    assert all("fusion_score" in r for r in rrf)


def test_fast_alignment_reranker():
    candidates = [
        {
            "chunk_id": "c1",
            "text": "General software engineering principles.",
            "title": "SE",
            "score": 0.8,
        },
        {
            "chunk_id": "c2",
            "text": "Mutual TLS encryption between microservices.",
            "title": "mTLS Guide",
            "score": 0.75,
        },
    ]
    reranker = FastAlignmentReranker()
    reranked = reranker.rerank("Mutual TLS encryption", candidates, top_k=2)
    assert len(reranked) == 2
    # c2 has exact phrase match and title match, should be ranked #1
    assert reranked[0]["chunk_id"] == "c2"
    assert "rerank_score" in reranked[0]
