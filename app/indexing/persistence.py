"""SQLite-backed metadata store, vector-to-chunk mappings, and index persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.ingestion.metadata import ChunkRecord, DocumentRecord


class MetadataStore:
    """ACID SQLite metadata storage mapping FAISS vector IDs to rich document/chunk metadata."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    language TEXT,
                    mime_type TEXT,
                    checksum TEXT,
                    tenant_id TEXT,
                    metadata_json TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    faiss_id INTEGER,
                    document_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    token_count INTEGER,
                    character_count INTEGER,
                    checksum TEXT,
                    tenant_id TEXT,
                    metadata_json TEXT,
                    is_tombstone INTEGER DEFAULT 0,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tombstone ON chunks(is_tombstone);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_tenant ON documents(tenant_id);")
            conn.commit()

    def add_documents(self, documents: list[DocumentRecord]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    d.document_id,
                    d.source,
                    d.title,
                    d.created_at,
                    d.updated_at,
                    d.language,
                    d.mime_type,
                    d.checksum,
                    d.tenant_id,
                    json.dumps(d.metadata),
                )
                for d in documents
            ]
            cursor.executemany(
                """
                INSERT OR REPLACE INTO documents
                (document_id, source, title, created_at, updated_at, language, mime_type, checksum, tenant_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def add_chunks(self, chunks_with_faiss_id: list[tuple[int, ChunkRecord]]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    chk.chunk_id,
                    faiss_id,
                    chk.document_id,
                    chk.chunk_index,
                    chk.text,
                    chk.token_count,
                    chk.character_count,
                    chk.checksum,
                    chk.tenant_id,
                    json.dumps(chk.metadata),
                    0,
                )
                for faiss_id, chk in chunks_with_faiss_id
            ]
            cursor.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, faiss_id, document_id, chunk_index, text, token_count, character_count, checksum, tenant_id, metadata_json, is_tombstone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def get_chunks_by_faiss_ids(self, faiss_ids: list[int]) -> dict[int, dict]:
        if not faiss_ids:
            return {}

        placeholders = ",".join("?" for _ in faiss_ids)
        query = f"""
            SELECT c.chunk_id, c.faiss_id, c.document_id, c.chunk_index, c.text, c.token_count,
                   c.character_count, c.tenant_id, c.metadata_json as chunk_meta, c.is_tombstone,
                   d.title, d.source, d.language, d.mime_type, d.metadata_json as doc_meta
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.document_id
            WHERE c.faiss_id IN ({placeholders}) AND c.is_tombstone = 0
        """  # nosec B608
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, faiss_ids)
            results = {}
            for row in cursor.fetchall():
                c_meta = json.loads(row["chunk_meta"] or "{}")
                d_meta = json.loads(row["doc_meta"] or "{}")
                merged_meta = {**d_meta, **c_meta}

                results[row["faiss_id"]] = {
                    "chunk_id": row["chunk_id"],
                    "faiss_id": row["faiss_id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "token_count": row["token_count"],
                    "character_count": row["character_count"],
                    "tenant_id": row["tenant_id"],
                    "title": row["title"] or "Untitled",
                    "source": row["source"] or "unknown",
                    "language": row["language"] or "en",
                    "mime_type": row["mime_type"] or "text/plain",
                    "metadata": merged_meta,
                }
            return results

    def get_document(self, document_id: str) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
            return data

    def list_documents(
        self, tenant_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id:
                cursor.execute(
                    "SELECT * FROM documents WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (tenant_id, limit, offset),
                )
            else:
                cursor.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            docs = []
            for row in cursor.fetchall():
                data = dict(row)
                data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
                docs.append(data)
            return docs

    def delete_document(self, document_id: str, tenant_id: str | None = None) -> bool:
        """Tombstone chunks and remove document for immediate exclusion from retrieval."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Verify document exists and matches tenant
            cursor.execute("SELECT tenant_id FROM documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                return False
            if tenant_id and row["tenant_id"] != tenant_id:
                return False

            cursor.execute(
                "UPDATE chunks SET is_tombstone = 1 WHERE document_id = ?", (document_id,)
            )
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return True

    def get_matching_faiss_ids(
        self,
        tenant_id: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> set[int] | None:
        """Return set of valid FAISS IDs matching tenant and optional metadata filter predicates."""
        where_clauses = ["c.is_tombstone = 0", "c.tenant_id = ?"]
        params: list[Any] = [tenant_id]

        if filters:
            for k, v in filters.items():
                if k in ("source", "language", "mime_type", "title"):
                    where_clauses.append(f"d.{k} = ?")
                    params.append(v)
                elif k == "category":
                    where_clauses.append("json_extract(d.metadata_json, '$.category') = ?")
                    params.append(v)
                elif k == "author":
                    where_clauses.append("json_extract(d.metadata_json, '$.author') = ?")
                    params.append(v)
                elif k == "created_after":
                    where_clauses.append("d.created_at >= ?")
                    params.append(v)
                elif k == "tags" and isinstance(v, (str, list)):
                    tag_str = v if isinstance(v, str) else json.dumps(v)
                    where_clauses.append("json_extract(d.metadata_json, '$.tags') LIKE ?")
                    params.append(f"%{tag_str}%")

        query = f"""
            SELECT c.faiss_id
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.document_id
            WHERE {" AND ".join(where_clauses)}
        """  # nosec B608

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return {r["faiss_id"] for r in rows}

    def count_documents(self, tenant_id: str | None = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE tenant_id = ?", (tenant_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def count_chunks(self, tenant_id: str | None = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM chunks WHERE tenant_id = ? AND is_tombstone = 0",
                    (tenant_id,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE is_tombstone = 0")
            return cursor.fetchone()[0]

    def get_all_active_chunks(self, tenant_id: str | None = None) -> list[dict]:
        """Fetch active chunks for building BM25 index."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id:
                cursor.execute(
                    "SELECT chunk_id, faiss_id, text, document_id FROM chunks WHERE tenant_id = ? AND is_tombstone = 0",
                    (tenant_id,),
                )
            else:
                cursor.execute(
                    "SELECT chunk_id, faiss_id, text, document_id FROM chunks WHERE is_tombstone = 0"
                )
            return [dict(r) for r in cursor.fetchall()]
