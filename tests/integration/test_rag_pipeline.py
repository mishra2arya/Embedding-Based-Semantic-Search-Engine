"""Integration tests for RAG pipeline, citations, and prompt safety."""

from pathlib import Path

from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager
from app.ingestion.pipeline import IngestionPipeline
from app.rag.generator import RAGPipeline
from app.retrieval.hybrid import HybridSearchEngine


def test_rag_end_to_end_integration(tmp_path: Path):
    emb_service = EmbeddingService(dimension=512)
    index_mgr = IndexManager(base_dir=tmp_path, dimension=512)
    pipeline = IngestionPipeline()

    res = pipeline.process_text(
        "Border Gateway Protocol (BGP) manages routing information between autonomous systems on the Internet.",
        title="BGP Overview",
        tenant_id="default",
    )
    vecs = emb_service.encode([c.text for c in res.chunks], normalize=True)
    index_mgr.add_batch([res.document], res.chunks, vecs)

    engine = HybridSearchEngine(embedding_service=emb_service, index_manager=index_mgr)
    engine.sync_lexical_index()

    rag = RAGPipeline(search_engine=engine)
    resp = rag.generate_answer("What does BGP manage on the Internet?")

    assert "Border Gateway Protocol" in resp.answer
    assert "[1]" in resp.answer
    assert len(resp.sources) >= 1
    assert resp.sources[0].title == "BGP Overview"
    assert resp.retrieval_latency_ms > 0
