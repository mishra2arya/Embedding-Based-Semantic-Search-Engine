"""Unit tests for FAISS index wrapper (HNSW, FlatIP, IVF)."""

import numpy as np

from app.indexing.faiss_index import FaissIndexWrapper


def test_faiss_hnsw_construction_and_search():
    wrapper = FaissIndexWrapper(dimension=512, index_type="HNSW")
    vecs = np.random.randn(10, 512).astype(np.float32)
    faiss_ids = np.arange(10, dtype=np.int64)

    wrapper.add(vecs, faiss_ids)
    assert wrapper.total_vectors == 10

    # Search top 3
    distances, indices = wrapper.search(vecs[0:1], top_k=3)
    assert indices.shape == (1, 3)
    assert indices[0][0] == 0  # Nearest neighbor should be itself


def test_faiss_flat_ip():
    wrapper = FaissIndexWrapper(dimension=512, index_type="FlatIP", metric="cosine")
    vecs = np.random.randn(5, 512).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    wrapper.add(vecs, np.arange(5, dtype=np.int64))
    assert wrapper.total_vectors == 5

    distances, indices = wrapper.search(vecs[2:3], top_k=1)
    assert indices[0][0] == 2
    assert abs(distances[0][0] - 1.0) < 1e-4
