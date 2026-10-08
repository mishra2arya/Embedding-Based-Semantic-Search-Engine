"""Lexical retrieval using Okapi BM25 and TF-IDF baseline."""

from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase tokenization removing non-alphanumeric punctuation."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in clean.split() if len(token) > 1]


class BM25Retriever:
    """Production lexical retriever using Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.faiss_ids: list[int] = []
        self.corpus_chunks: list[dict] = []

    def fit(self, chunks: list[dict]) -> None:
        """Build BM25 index over collection of chunk dictionaries."""
        if not chunks:
            self.bm25 = None
            self.chunk_ids = []
            self.faiss_ids = []
            self.corpus_chunks = []
            return

        self.corpus_chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.faiss_ids = [c.get("faiss_id", idx) for idx, c in enumerate(chunks)]

        tokenized_corpus = [simple_tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """Query BM25 index returning scored chunk candidates."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        query_set = set(query_tokens)
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            text_tokens = set(simple_tokenize(self.corpus_chunks[idx].get("text", "")))
            has_overlap = bool(query_set & text_tokens)

            # In small test corpora, BM25 IDF can mathematically equal 0.0
            if score > 0.0 or has_overlap:
                chunk = dict(self.corpus_chunks[idx])
                effective_score = max(score, 0.1 if has_overlap else 0.0)
                chunk["lexical_score"] = effective_score
                chunk["score"] = effective_score
                results.append(chunk)

        return results


class TfidfRetriever:
    """Baseline lexical retriever using TF-IDF for benchmarking."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=simple_tokenize, token_pattern=None, max_features=50000
        )
        self.tfidf_matrix = None
        self.corpus_chunks: list[dict] = []

    def fit(self, chunks: list[dict]) -> None:
        if not chunks:
            self.tfidf_matrix = None
            self.corpus_chunks = []
            return

        self.corpus_chunks = chunks
        texts = [c.get("text", "") for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        if self.tfidf_matrix is None or not self.corpus_chunks:
            return []

        q_vec = self.vectorizer.transform([query])
        # Compute cosine similarity
        similarities = (self.tfidf_matrix * q_vec.T).toarray().flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0.0:
                continue
            chunk = dict(self.corpus_chunks[idx])
            chunk["lexical_score"] = score
            chunk["score"] = score
            results.append(chunk)

        return results
