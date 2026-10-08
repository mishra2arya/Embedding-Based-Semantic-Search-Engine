"""Information retrieval evaluation metrics: Precision@K, Recall@K, MRR@K, nDCG@K, HitRate@K."""

from __future__ import annotations

import math


class RetrievalMetrics:
    """Calculates standard Information Retrieval (IR) evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Fraction of retrieved documents in top-K that are relevant."""
        if k <= 0 or not retrieved_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Fraction of relevant documents that are successfully retrieved in top-K."""
        if not relevant_ids or k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / float(len(relevant_ids))

    @staticmethod
    def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Binary indicator (1.0 or 0.0) whether at least one relevant document is in top-K."""
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

    @staticmethod
    def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Mean Reciprocal Rank: reciprocal of the rank of the first relevant document."""
        top_k = retrieved_ids[:k]
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain at rank K."""
        top_k = retrieved_ids[:k]
        if not top_k or not relevant_ids:
            return 0.0

        # Compute DCG
        dcg = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1)

        # Compute IDCG (Ideal DCG)
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum((2.0**1.0 - 1.0) / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        if idcg <= 0.0:
            return 0.0
        return dcg / idcg
