"""Production index builder creating FAISS index, SQLite metadata store, and manifest."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.core.config import settings
from app.embeddings.model import EmbeddingService
from app.indexing.faiss_index import FaissIndexWrapper
from app.indexing.persistence import MetadataStore
from app.indexing.versioning import IndexVersionManager
from app.ingestion.metadata import ChunkRecord, DocumentRecord
from app.retrieval.lexical import BM25Retriever


def build_production_index(
    processed_dir: Path | str = "./data/processed",
    output_index_dir: Path | str = "./data/indexes",
    index_type: str = "HNSW",
    dimension: int = 512,
    batch_size: int = 256,
    max_vectors: int = 0,
) -> None:
    proc_path = Path(processed_dir)
    idx_base = Path(output_index_dir)
    idx_base.mkdir(parents=True, exist_ok=True)

    version_mgr = IndexVersionManager(idx_base)
    version_name = version_mgr.get_next_version_name()
    version_dir = version_mgr.get_version_dir(version_name)
    version_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building index version '{version_name}' in '{version_dir}'...")
    start_time = time.time()

    # 1. Initialize components
    embedding_service = EmbeddingService(dimension=dimension, batch_size=batch_size)
    faiss_wrapper = FaissIndexWrapper(
        dimension=dimension,
        index_type=index_type,
        metric=settings.faiss_metric,
        m=settings.faiss_m,
        ef_construction=settings.faiss_ef_construction,
        ef_search=settings.faiss_ef_search,
        nlist=settings.faiss_nlist,
        nprobe=settings.faiss_nprobe,
    )
    metadata_store = MetadataStore(version_dir / "metadata.db")

    # 2. Find processed chunk files
    chunk_files = sorted(list(proc_path.glob("processed_chunks_part_*.jsonl")))
    doc_files = sorted(list(proc_path.glob("processed_docs_part_*.jsonl")))

    if not chunk_files:
        print(
            f"No processed chunk files found in '{proc_path}'. Run 'python scripts/ingest.py' first."
        )
        return

    # Ingest document records into metadata store first
    print("Loading document metadata...")
    total_docs = 0
    for doc_file in doc_files:
        docs = []
        with open(doc_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(DocumentRecord(**json.loads(line)))
        if docs:
            metadata_store.add_documents(docs)
            total_docs += len(docs)
    print(f"Loaded {total_docs:,} document records into metadata store.")

    # Ingest chunks, encode vectors, and add to FAISS
    total_vectors = 0
    chunk_batch: list[ChunkRecord] = []
    bm25_corpus: list[dict] = []

    for c_file in chunk_files:
        if max_vectors > 0 and total_vectors >= max_vectors:
            break

        with open(c_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_record = ChunkRecord(**record)
                chunk_batch.append(chunk_record)

                if len(chunk_batch) >= batch_size:
                    texts = [c.text for c in chunk_batch]
                    vectors = embedding_service.encode(texts, normalize=True, use_cache=False)

                    start_faiss_id = total_vectors
                    faiss_ids = np.arange(
                        start_faiss_id, start_faiss_id + len(chunk_batch), dtype=np.int64
                    )

                    faiss_wrapper.add(vectors, faiss_ids)
                    chunks_with_ids = list(zip(faiss_ids.tolist(), chunk_batch, strict=False))
                    metadata_store.add_chunks(chunks_with_ids)

                    # Collect for BM25
                    for fid, c in chunks_with_ids:
                        bm25_corpus.append(
                            {
                                "chunk_id": c.chunk_id,
                                "faiss_id": fid,
                                "text": c.text,
                                "tenant_id": c.tenant_id,
                            }
                        )

                    total_vectors += len(chunk_batch)
                    chunk_batch = []

                    elapsed = time.time() - start_time
                    rate = total_vectors / elapsed if elapsed > 0 else 0
                    print(f"  Indexed: {total_vectors:,} vectors ({rate:,.0f} vectors/sec)")

                    if max_vectors > 0 and total_vectors >= max_vectors:
                        break

    if chunk_batch:
        texts = [c.text for c in chunk_batch]
        vectors = embedding_service.encode(texts, normalize=True, use_cache=False)
        start_faiss_id = total_vectors
        faiss_ids = np.arange(start_faiss_id, start_faiss_id + len(chunk_batch), dtype=np.int64)
        faiss_wrapper.add(vectors, faiss_ids)
        chunks_with_ids = list(zip(faiss_ids.tolist(), chunk_batch, strict=False))
        metadata_store.add_chunks(chunks_with_ids)
        for fid, c in chunks_with_ids:
            bm25_corpus.append(
                {"chunk_id": c.chunk_id, "faiss_id": fid, "text": c.text, "tenant_id": c.tenant_id}
            )
        total_vectors += len(chunk_batch)

    # 3. Save FAISS index
    index_file = version_dir / "index.faiss"
    print(f"Serializing FAISS index ({total_vectors:,} vectors) to '{index_file}'...")
    faiss_wrapper.save(str(index_file))

    # 4. Fit and persist BM25 lexical index
    print(f"Fitting BM25 lexical index over {len(bm25_corpus):,} chunks...")
    bm25 = BM25Retriever()
    bm25.fit(bm25_corpus)
    with open(version_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Write manifest & checksum
    version_mgr.write_manifest_and_checksum(
        version_dir=version_dir,
        dimension=dimension,
        index_type=index_type,
        metric=settings.faiss_metric,
        documents_count=total_docs,
        vectors_count=total_vectors,
        model_name=embedding_service.model_name,
    )
    print("Manifest and SHA-256 checksums generated.")

    # 6. Validate & Activate
    is_valid = version_mgr.validate_version(version_dir, expected_dimension=dimension)
    if not is_valid:
        raise RuntimeError(f"Validation failed for newly built version '{version_name}'.")

    version_mgr.activate_version(version_name, expected_dimension=dimension)
    build_time = time.time() - start_time
    print(
        f"SUCCESS: Built and activated index version '{version_name}' with {total_vectors:,} vectors "
        f"and {total_docs:,} documents in {build_time:.1f}s ({total_vectors / build_time:,.0f} vectors/sec)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build production FAISS and metadata index")
    parser.add_argument(
        "--input", type=str, default="./data/processed", help="Directory of processed chunks"
    )
    parser.add_argument(
        "--output", type=str, default="./data/indexes", help="Base indexes directory"
    )
    parser.add_argument(
        "--index-type", type=str, default="HNSW", help="FAISS index type (HNSW, IVF, FlatIP)"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Vector encoding batch size")
    parser.add_argument(
        "--max-vectors", type=int, default=0, help="Optional max vector cap (0 = all)"
    )
    args = parser.parse_args()

    build_production_index(
        processed_dir=args.input,
        output_index_dir=args.output,
        index_type=args.index_type,
        batch_size=args.batch_size,
        max_vectors=args.max_vectors,
    )
