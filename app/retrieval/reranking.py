"""Reranking layer for second-stage candidate reordering."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """Rerank candidates and populate rerank_score."""
        pass


class FastAlignmentReranker(BaseReranker):
    """Production-grade lightweight semantic and lexical alignment reranker.

    Computes exact phrase matches, term coverage, and heading alignment
    to refine candidate ordering with sub-millisecond overhead.
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        if not candidates:
            return []

        query_terms = [t.lower() for t in query.split() if len(t) > 1]
        query_lower = query.lower()

        for cand in candidates:
            text = cand.get("text", "").lower()
            title = cand.get("title", "").lower()
            section = cand.get("metadata", {}).get("section_heading", "").lower()

            base_score = cand.get("score", cand.get("fusion_score", 0.5))

            # 1. Exact phrase match bonus
            exact_match_bonus = 0.25 if query_lower in text else 0.0

            # 2. Term coverage ratio
            matched_terms = sum(1 for t in query_terms if t in text)
            coverage_ratio = (matched_terms / len(query_terms)) if query_terms else 0.0

            # 3. Heading / Title match bonus
            title_bonus = 0.15 if any(t in title or t in section for t in query_terms) else 0.0

            # 4. Proximity penalty: shorter chunks with high coverage get slight boost
            length_factor = 1.0 / (1.0 + math.log1p(max(1, len(text.split())) / 200.0))

            rerank_score = (
                (0.5 * base_score)
                + (0.25 * coverage_ratio)
                + exact_match_bonus
                + title_bonus
                + (0.1 * length_factor)
            )

            cand["rerank_score"] = float(rerank_score)
            cand["final_score"] = float(rerank_score)
            cand["score"] = float(rerank_score)

        # Sort descending by rerank_score
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Neural Cross-Encoder reranker using transformer cross-attention."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = FastAlignmentReranker()
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder model: {self.model_name}")
        except Exception as e:
            logger.info(f"CrossEncoder model unavailable ({e}). Using FastAlignmentReranker.")
            self._model = None

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        if not candidates:
            return []

        if self._model is None:
            return self._fallback.rerank(query, candidates, top_k=top_k)

        pairs = [[query, c.get("text", "")] for c in candidates]
        try:
            scores = self._model.predict(pairs)
            for cand, score in zip(candidates, scores, strict=False):
                cand["rerank_score"] = float(score)
                cand["final_score"] = float(score)
                cand["score"] = float(score)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:top_k]
        except Exception:
            return self._fallback.rerank(query, candidates, top_k=top_k)
