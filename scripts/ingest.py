"""High-throughput streaming document ingestion script."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.file_loader import RawDocument
from app.ingestion.pipeline import IngestionPipeline


def run_ingestion(
    input_dir: Path | str = "./data/raw",
    output_dir: Path | str = "./data/processed",
    max_docs: int = 0,
    batch_size: int = 10000,
) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline = IngestionPipeline()
    raw_files = sorted(list(in_path.glob("*.jsonl")) + list(in_path.glob("*.txt")))

    if not raw_files:
        print(
            f"No document files found in '{in_path}'. Run 'python scripts/generate_dataset.py' first."
        )
        return

    print(
        f"Found {len(raw_files)} raw files in '{in_path}'. Starting ingestion into '{out_path}'..."
    )
    start_time = time.time()

    total_processed = 0
    total_duplicates = 0
    total_chunks = 0
    out_file_idx = 1

    current_chunk_batch: list[dict] = []
    current_doc_batch: list[dict] = []

    def flush_batch(idx: int):
        nonlocal current_chunk_batch, current_doc_batch
        if not current_chunk_batch and not current_doc_batch:
            return
        doc_out = out_path / f"processed_docs_part_{idx:03d}.jsonl"
        chk_out = out_path / f"processed_chunks_part_{idx:03d}.jsonl"

        with open(doc_out, "w", encoding="utf-8") as f:
            for d in current_doc_batch:
                f.write(json.dumps(d) + "\n")

        with open(chk_out, "w", encoding="utf-8") as f:
            for c in current_chunk_batch:
                f.write(json.dumps(c) + "\n")

        current_doc_batch = []
        current_chunk_batch = []

    for file_path in raw_files:
        if max_docs > 0 and total_processed >= max_docs:
            break

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    raw_doc = RawDocument(
                        content=record.get("text", "").encode("utf-8"),
                        filename=f"{record.get('id', 'doc')}.txt",
                        source=record.get("source", "raw_ingest"),
                        tenant_id=record.get("tenant_id", "default"),
                        metadata=record.get("metadata", {}),
                    )
                    res = pipeline.process_raw_document(raw_doc)
                    res.document.title = record.get("title", res.document.title)

                    total_processed += 1

                    if res.is_duplicate:
                        total_duplicates += 1
                        continue

                    current_doc_batch.append(res.document.model_dump())
                    for c in res.chunks:
                        current_chunk_batch.append(c.model_dump())
                        total_chunks += 1

                    if len(current_chunk_batch) >= batch_size:
                        flush_batch(out_file_idx)
                        out_file_idx += 1
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        print(
                            f"  Ingested: {total_processed:,} docs ({total_duplicates:,} dups skipped, {total_chunks:,} chunks) - {rate:,.0f} docs/sec"
                        )

                    if max_docs > 0 and total_processed >= max_docs:
                        break

                except Exception:
                    continue

    flush_batch(out_file_idx)
    total_time = time.time() - start_time
    print(
        f"Ingestion complete: {total_processed:,} documents ingested ({total_duplicates:,} duplicates skipped, "
        f"{total_chunks:,} chunks created) in {total_time:.1f}s ({total_processed / total_time:,.0f} docs/sec)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk raw document corpus")
    parser.add_argument(
        "--input", type=str, default="./data/raw", help="Directory containing raw JSONL files"
    )
    parser.add_argument(
        "--output", type=str, default="./data/processed", help="Directory for processed chunks"
    )
    parser.add_argument("--max-docs", type=int, default=0, help="Optional document cap (0 = all)")
    args = parser.parse_args()

    run_ingestion(input_dir=args.input, output_dir=args.output, max_docs=args.max_docs)
