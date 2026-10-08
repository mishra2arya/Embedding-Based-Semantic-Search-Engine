"""Hybrid retrieval orchestrator uniting dense, lexical, fusion, and reranking layers."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import settings
from app.core.security import validate_query_string
from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager
from app.retrieval.filtering import MetadataFilter
from app.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.lexical import BM25Retriever
from app.retrieval.reranking import BaseReranker, FastAlignmentReranker
from app.retrieval.semantic import SemanticRetriever

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Production hybrid retrieval engine returning explainable retrieval diagnostics."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_manager: IndexManager,
        lexical_retriever: BM25Retriever | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.embedding_service = embedding_service
        self.index_manager = index_manager
        self.semantic_retriever = SemanticRetriever(embedding_service, index_manager)
        self.lexical_retriever = lexical_retriever or BM25Retriever()
        self.reranker = reranker or FastAlignmentReranker()

    def sync_lexical_index(self, tenant_id: str | None = None) -> None:
        """Sync active chunks from metadata store into BM25 retriever."""
        if self.index_manager.is_ready() and self.index_manager.metadata_store:
            chunks = self.index_manager.metadata_store.get_all_active_chunks(tenant_id=tenant_id)
            self.lexical_retriever.fit(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        tenant_id: str = "default",
        mode: str = "hybrid",  # "hybrid", "semantic", "lexical"
        rerank: bool = False,
        fusion_method: str = "weighted",  # "weighted" or "rrf"
        candidate_k: int = 50,
    ) -> dict:
        """Execute search returning explainable results and granular latency profile."""
        total_start = time.perf_counter()
        latencies: dict[str, float] = {}

        # 1. Query Preprocessing & Validation
        t0 = time.perf_counter()
        clean_query = validate_query_string(query, max_length=settings.max_query_length)
        latencies["query_preprocessing_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Dense Semantic Retrieval
        semantic_candidates: list[dict] = []
        if mode in ("hybrid", "semantic"):
            t_emb = time.perf_counter()
            query_vec = self.embedding_service.encode(clean_query, normalize=True, use_cache=True)
            latencies["embedding_generation_ms"] = (time.perf_counter() - t_emb) * 1000.0

            t_faiss = time.perf_counter()
            semantic_candidates = self.index_manager.search(
                query_vector=query_vec,
                top_k=candidate_k,
                tenant_id=tenant_id,
                filters=filters,
                candidate_k=candidate_k,
            )
            latencies["faiss_retrieval_ms"] = (time.perf_counter() - t_faiss) * 1000.0
        else:
            latencies["embedding_generation_ms"] = 0.0
            latencies["faiss_retrieval_ms"] = 0.0

        # 3. Lexical Retrieval
        lexical_candidates: list[dict] = []
        if mode in ("hybrid", "lexical"):
            t_lex = time.perf_counter()
            raw_lex = self.lexical_retriever.search(clean_query, top_k=candidate_k)
            # Filter lexical candidates by metadata and tenant
            lexical_candidates = MetadataFilter.filter_candidates(
                raw_lex, filters=filters, tenant_id=tenant_id
            )
            latencies["lexical_retrieval_ms"] = (time.perf_counter() - t_lex) * 1000.0
        else:
            latencies["lexical_retrieval_ms"] = 0.0

        # 4. Candidate Fusion
        t_fuse = time.perf_counter()
        if mode == "semantic":
            fused_candidates = semantic_candidates[:candidate_k]
            for c in fused_candidates:
                c["fusion_score"] = c.get("semantic_score", 0.0)
                c["final_score"] = c["fusion_score"]
        elif mode == "lexical":
            fused_candidates = lexical_candidates[:candidate_k]
            for c in fused_candidates:
                c["fusion_score"] = c.get("lexical_score", 0.0)
                c["final_score"] = c["fusion_score"]
        else:
            # Hybrid fusion
            if fusion_method == "rrf":
                fused_candidates = reciprocal_rank_fusion(
                    semantic_results=semantic_candidates,
                    lexical_results=lexical_candidates,
                    rrf_k=settings.retrieval_rrf_k,
                    top_k=candidate_k,
                )
            else:
                fused_candidates = weighted_score_fusion(
                    semantic_results=semantic_candidates,
                    lexical_results=lexical_candidates,
                    semantic_weight=settings.retrieval_semantic_weight,
                    lexical_weight=settings.retrieval_lexical_weight,
                    top_k=candidate_k,
                )
        latencies["candidate_fusion_ms"] = (time.perf_counter() - t_fuse) * 1000.0

        # 5. Metadata Post-Filtering Verification
        t_filter = time.perf_counter()
        valid_candidates = MetadataFilter.filter_candidates(
            fused_candidates, filters=filters, tenant_id=tenant_id
        )
        latencies["metadata_filtering_ms"] = (time.perf_counter() - t_filter) * 1000.0

        # 6. Optional Reranking
        if rerank and self.reranker:
            t_rerank = time.perf_counter()
            final_results = self.reranker.rerank(clean_query, valid_candidates, top_k=top_k)
            latencies["reranking_ms"] = (time.perf_counter() - t_rerank) * 1000.0
        else:
            latencies["reranking_ms"] = 0.0
            final_results = valid_candidates[:top_k]
            for c in final_results:
                c["rerank_score"] = None
                c["final_score"] = c.get("fusion_score", c.get("score", 0.0))

        # 7. Citations & Rank Formatting
        formatted_results = []
        for rank, item in enumerate(final_results, start=1):
            doc_id = item.get("document_id", "unknown")
            chunk_id = item.get("chunk_id", "unknown")
            title = item.get("title", "Untitled")

            citation = f'[{rank}] "{title}" (Doc: {doc_id}, Chunk: {chunk_id})'

            formatted_results.append(
                {
                    "rank": rank,
                    "score": round(float(item.get("final_score", item.get("score", 0.0))), 4),
                    "semantic_score": round(float(item.get("semantic_score", 0.0)), 4)
                    if item.get("semantic_score") is not None
                    else None,
                    "lexical_score": round(float(item.get("lexical_score", 0.0)), 4)
                    if item.get("lexical_score") is not None
                    else None,
                    "fusion_score": round(float(item.get("fusion_score", 0.0)), 4)
                    if item.get("fusion_score") is not None
                    else None,
                    "rerank_score": round(float(item["rerank_score"]), 4)
                    if item.get("rerank_score") is not None
                    else None,
                    "final_score": round(float(item.get("final_score", item.get("score", 0.0))), 4),
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "citation": citation,
                }
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        latencies["total_ms"] = round(total_ms, 2)

        return {
            "query": clean_query,
            "mode": mode,
            "reranked": rerank,
            "total_candidates": len(valid_candidates),
            "results": formatted_results,
            "latency_ms": latencies["total_ms"],
            "latency_profile": {k: round(v, 2) for k, v in latencies.items()},
        }
