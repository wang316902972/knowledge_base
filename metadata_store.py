"""SQLite metadata store for updatable FAISS knowledge bases.

FAISS stores vectors efficiently, but it should not be the source of truth for
chunk lifecycle state. This module keeps stable vector IDs and active/deleted
state in SQLite so updates can be append-only and safe at million-chunk scale.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class SQLiteMetadataStore:
    """Stores chunk metadata and lifecycle state with stable integer IDs."""

    def __init__(
        self,
        db_path: str | Path,
        compaction_deleted_ratio: float = 0.3,
    ) -> None:
        self.db_path = Path(db_path)
        self.compaction_deleted_ratio = compaction_deleted_ratio
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                source_document TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_active_vector ON chunks(active, vector_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_hash_active ON chunks(text_hash, active)"
        )
        self.conn.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_chunks(
        self,
        texts: Sequence[str],
        source_document: Optional[str] = None,
    ) -> List[int]:
        """Append active chunks and return their stable vector IDs."""
        if not texts:
            return []

        now = time.time()
        vector_ids: List[int] = []
        with self.conn:
            for text in texts:
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks
                        (text, text_hash, source_document, active, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (text, self._hash_text(text), source_document, now, now),
                )
                vector_ids.append(int(cursor.lastrowid))
        return vector_ids

    def add_chunks_with_ids(
        self,
        chunks: Sequence[Tuple[int, str]],
        source_document: Optional[str] = None,
    ) -> List[int]:
        """Append active chunks with caller-provided stable vector IDs."""
        if not chunks:
            return []

        now = time.time()
        vector_ids: List[int] = []
        with self.conn:
            for vector_id, text in chunks:
                self.conn.execute(
                    """
                    INSERT INTO chunks
                        (vector_id, text, text_hash, source_document, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (int(vector_id), text, self._hash_text(text), source_document, now, now),
                )
                vector_ids.append(int(vector_id))
        return vector_ids

    def import_active_chunks(
        self,
        chunks: Iterable[Tuple[int, str]],
        source_document: Optional[str] = "legacy-json",
    ) -> None:
        """Import legacy vector IDs without changing their values."""
        now = time.time()
        with self.conn:
            for vector_id, text in chunks:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO chunks
                        (vector_id, text, text_hash, source_document, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (int(vector_id), text, self._hash_text(text), source_document, now, now),
                )

    def soft_delete_texts(self, texts: Sequence[str]) -> List[int]:
        """Mark all active chunks with exact matching text inactive."""
        deleted_ids: List[int] = []
        now = time.time()
        with self.conn:
            for text in texts:
                rows = self.conn.execute(
                    """
                    SELECT vector_id
                    FROM chunks
                    WHERE active = 1 AND text_hash = ? AND text = ?
                    ORDER BY vector_id
                    """,
                    (self._hash_text(text), text),
                ).fetchall()
                ids = [int(row["vector_id"]) for row in rows]
                if not ids:
                    continue

                placeholders = ",".join("?" for _ in ids)
                self.conn.execute(
                    f"""
                    UPDATE chunks
                    SET active = 0, updated_at = ?, deleted_at = ?
                    WHERE vector_id IN ({placeholders})
                    """,
                    [now, now, *ids],
                )
                deleted_ids.extend(ids)
        return deleted_ids

    def get_active_ids_for_text(self, text: str) -> List[int]:
        rows = self.conn.execute(
            """
            SELECT vector_id
            FROM chunks
            WHERE active = 1 AND text_hash = ? AND text = ?
            ORDER BY vector_id
            """,
            (self._hash_text(text), text),
        ).fetchall()
        return [int(row["vector_id"]) for row in rows]

    def soft_delete_vector_ids(self, vector_ids: Sequence[int]) -> List[int]:
        """Mark active vector IDs inactive."""
        ids = [int(vector_id) for vector_id in vector_ids]
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT vector_id
            FROM chunks
            WHERE active = 1 AND vector_id IN ({placeholders})
            ORDER BY vector_id
            """,
            ids,
        ).fetchall()
        active_ids = [int(row["vector_id"]) for row in rows]
        if not active_ids:
            return []

        now = time.time()
        placeholders = ",".join("?" for _ in active_ids)
        with self.conn:
            self.conn.execute(
                f"""
                UPDATE chunks
                SET active = 0, updated_at = ?, deleted_at = ?
                WHERE vector_id IN ({placeholders})
                """,
                [now, now, *active_ids],
            )
        return active_ids

    def get_text(self, vector_id: int, include_deleted: bool = False) -> Optional[str]:
        condition = "" if include_deleted else "AND active = 1"
        row = self.conn.execute(
            f"SELECT text FROM chunks WHERE vector_id = ? {condition}",
            (int(vector_id),),
        ).fetchone()
        return str(row["text"]) if row else None

    def is_active(self, vector_id: int) -> bool:
        row = self.conn.execute(
            "SELECT active FROM chunks WHERE vector_id = ?",
            (int(vector_id),),
        ).fetchone()
        return bool(row["active"]) if row else False

    def get_active_vector_ids(self) -> List[int]:
        rows = self.conn.execute(
            "SELECT vector_id FROM chunks WHERE active = 1 ORDER BY vector_id"
        ).fetchall()
        return [int(row["vector_id"]) for row in rows]

    def get_active_chunks(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT vector_id, text, source_document
            FROM chunks
            WHERE active = 1
            ORDER BY vector_id
            """
        ).fetchall()
        return [
            {
                "vector_id": int(row["vector_id"]),
                "text": str(row["text"]),
                "source_document": row["source_document"],
            }
            for row in rows
        ]

    def get_active_text_map(self) -> Dict[str, str]:
        return {
            str(chunk["vector_id"]): chunk["text"]
            for chunk in self.get_active_chunks()
        }

    def get_chunk_to_id_map(self) -> Dict[str, str]:
        rows = self.conn.execute(
            """
            SELECT text, MIN(vector_id) AS vector_id
            FROM chunks
            WHERE active = 1
            GROUP BY text
            """
        ).fetchall()
        return {str(row["text"]): str(int(row["vector_id"])) for row in rows}

    def replace_with_active_chunks(self, chunks: Sequence[Tuple[int, str]]) -> None:
        """Replace metadata with a compacted active-only snapshot."""
        now = time.time()
        with self.conn:
            self.conn.execute("DELETE FROM chunks")
            for vector_id, text in chunks:
                self.conn.execute(
                    """
                    INSERT INTO chunks
                        (vector_id, text, text_hash, source_document, active, created_at, updated_at)
                    VALUES (?, ?, ?, 'compacted', 1, ?, ?)
                    """,
                    (int(vector_id), text, self._hash_text(text), now, now),
                )

    def get_metrics(self) -> Dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_chunks,
                SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_chunks,
                SUM(CASE WHEN active = 0 THEN 1 ELSE 0 END) AS deleted_chunks
            FROM chunks
            """
        ).fetchone()
        total_chunks = int(row["total_chunks"] or 0)
        active_chunks = int(row["active_chunks"] or 0)
        deleted_chunks = int(row["deleted_chunks"] or 0)
        deleted_ratio = deleted_chunks / total_chunks if total_chunks else 0.0
        return {
            "total_chunks": total_chunks,
            "active_chunks": active_chunks,
            "deleted_chunks": deleted_chunks,
            "deleted_ratio": deleted_ratio,
            "needs_compaction": deleted_ratio >= self.compaction_deleted_ratio,
        }

    def close(self) -> None:
        self.conn.close()
