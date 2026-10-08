"""Unit tests for embedding model service, dimension validation, and caching."""

import numpy as np

from app.embeddings.batching import chunk_into_batches
from app.embeddings.cache import EmbeddingCache
from app.embeddings.model import EmbeddingService


def test_embedding_batching():
    items = list(range(250))
    batches = list(chunk_into_batches(items, batch_size=64))
    assert len(batches) == 4
    assert len(batches[0]) == 64
    assert len(batches[-1]) == 250 - (64 * 3)


def test_embedding_cache():
    cache = EmbeddingCache(max_size=3)
    vec1 = np.ones(512, dtype=np.float32)
    cache.put("Query One", vec1)

    cached = cache.get("Query One")
    assert cached is not None
    assert np.allclose(cached, vec1)
    assert cache.hits == 1

    # Fill and test eviction
    cache.put("Query Two", vec1)
    cache.put("Query Three", vec1)
    cache.put("Query Four", vec1)  # Evicts Query One
    assert cache.get("Query One") is None


def test_embedding_service_dimension_and_norm():
    service = EmbeddingService(dimension=512, batch_size=16)
    texts = [
        "First sentence for vector encoding.",
        "Second sentence testing normalization.",
    ]
    vectors = service.encode(texts, normalize=True)

    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32

    # Verify L2 norm
    for v in vectors:
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 1e-3
