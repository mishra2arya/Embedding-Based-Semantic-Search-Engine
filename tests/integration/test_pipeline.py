"""Integration tests for ingestion, embedding, FAISS indexing, and hybrid search."""

from pathlib import Path

from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid import HybridSearchEngine


def test_full_search_integration(tmp_path: Path):
    pipeline = IngestionPipeline()
    emb_service = EmbeddingService(dimension=512)
    index_mgr = IndexManager(base_dir=tmp_path, dimension=512)

    # Ingest 3 documents
    doc_texts = [
        (
            "Zero Trust Architecture",
            "Zero trust models require continuous authentication across all access points.",
        ),
        (
            "FAISS Indexing Guide",
            "Hierarchical Navigable Small World graphs provide approximate nearest neighbor retrieval.",
        ),
        (
            "Kubernetes Pod Scaling",
            "Horizontal Pod Autoscalers continuously adjust replica counts based on metrics.",
        ),
    ]

    for title, text in doc_texts:
        res = pipeline.process_text(text, title=title, tenant_id="default")
        vectors = emb_service.encode([c.text for c in res.chunks], normalize=True)
        index_mgr.add_batch([res.document], res.chunks, vectors)

    engine = HybridSearchEngine(embedding_service=emb_service, index_manager=index_mgr)
    engine.sync_lexical_index()

    # Query
    search_out = engine.search("How does zero trust authenticate users?", top_k=2, rerank=True)
    assert len(search_out["results"]) > 0
    top = search_out["results"][0]
    assert "Zero Trust" in top["title"]
    assert top["score"] > 0
    assert "total_ms" in search_out["latency_profile"]
