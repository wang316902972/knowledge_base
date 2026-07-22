"""SQLite metadata store for updatable FAISS knowledge bases.

FAISS stores vectors efficiently, but it should not be the source of truth for
chunk lifecycle state. This module keeps stable vector IDs and active/deleted
state in SQLite so updates can be append-only and safe at million-chunk scale.
"""

from __future__ import annotations

import hashlib
import json
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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_state (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO metadata_state(key, value)
            VALUES ('chunks_revision', 0)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sql_templates (
                external_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                template_id TEXT NOT NULL,
                intent_key TEXT NOT NULL,
                canonical_template TEXT NOT NULL,
                search_text TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                schema_fingerprint TEXT NOT NULL,
                template_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                reuse_count INTEGER NOT NULL DEFAULT 0,
                shadow_match_count INTEGER NOT NULL DEFAULT 0,
                validation_failure_count INTEGER NOT NULL DEFAULT 0,
                vector_id INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_used_at REAL,
                UNIQUE(dataset_id, template_id, template_version)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sql_templates_lookup
            ON sql_templates(
                dataset_id,
                status,
                intent_key,
                canonical_template
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sql_templates_vector
            ON sql_templates(vector_id)
            """
        )
        self.conn.commit()

    def _bump_chunks_revision(self) -> None:
        self.conn.execute(
            """
            UPDATE metadata_state
            SET value = value + 1
            WHERE key = 'chunks_revision'
            """
        )

    def get_chunks_revision(self) -> int:
        """Return the durable generation of the active chunk corpus."""
        row = self.conn.execute(
            "SELECT value FROM metadata_state WHERE key = 'chunks_revision'"
        ).fetchone()
        return int(row["value"]) if row else 0

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
            self._bump_chunks_revision()
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
            self._bump_chunks_revision()
        return vector_ids

    def import_active_chunks(
        self,
        chunks: Iterable[Tuple[int, str]],
        source_document: Optional[str] = "legacy-json",
    ) -> None:
        """Import legacy vector IDs without changing their values."""
        now = time.time()
        imported = False
        with self.conn:
            for vector_id, text in chunks:
                cursor = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO chunks
                        (vector_id, text, text_hash, source_document, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (int(vector_id), text, self._hash_text(text), source_document, now, now),
                )
                imported = imported or cursor.rowcount > 0
            if imported:
                self._bump_chunks_revision()

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
            if deleted_ids:
                self._bump_chunks_revision()
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
            self._bump_chunks_revision()
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

    def get_all_vector_ids(self) -> List[int]:
        rows = self.conn.execute(
            "SELECT vector_id FROM chunks ORDER BY vector_id"
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
            self._bump_chunks_revision()

    @staticmethod
    def _template_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        payload = json.loads(str(row["payload_json"]))
        payload.update(
            {
                "external_id": str(row["external_id"]),
                "dataset_id": str(row["dataset_id"]),
                "template_id": str(row["template_id"]),
                "intent_key": str(row["intent_key"]),
                "canonical_template": str(row["canonical_template"]),
                "search_text": str(row["search_text"]),
                "schema_fingerprint": str(row["schema_fingerprint"]),
                "template_version": int(row["template_version"]),
                "status": str(row["status"]),
                "source": str(row["source"]),
                "success_count": int(row["success_count"]),
                "reuse_count": int(row["reuse_count"]),
                "shadow_match_count": int(row["shadow_match_count"]),
                "validation_failure_count": int(
                    row["validation_failure_count"]
                ),
                "vector_id": (
                    int(row["vector_id"])
                    if row["vector_id"] is not None
                    else None
                ),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "last_used_at": (
                    float(row["last_used_at"])
                    if row["last_used_at"] is not None
                    else None
                ),
            }
        )
        return payload

    def upsert_sql_template(
        self,
        record: Dict[str, Any],
        vector_id: Optional[int],
    ) -> Dict[str, Any]:
        """Insert or update one structured SQL template by external_id."""
        now = time.time()
        existing = self.get_sql_template(str(record["external_id"]))
        payload = dict(record)
        for key in (
            "external_id",
            "dataset_id",
            "template_id",
            "intent_key",
            "canonical_template",
            "search_text",
            "schema_fingerprint",
            "template_version",
            "status",
            "source",
            "success_count",
            "reuse_count",
            "shadow_match_count",
            "validation_failure_count",
            "vector_id",
            "created_at",
            "updated_at",
            "last_used_at",
        ):
            payload.pop(key, None)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sql_templates (
                    external_id,
                    dataset_id,
                    template_id,
                    intent_key,
                    canonical_template,
                    search_text,
                    payload_json,
                    schema_fingerprint,
                    template_version,
                    status,
                    source,
                    success_count,
                    reuse_count,
                    shadow_match_count,
                    validation_failure_count,
                    vector_id,
                    created_at,
                    updated_at,
                    last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(external_id) DO UPDATE SET
                    dataset_id = excluded.dataset_id,
                    template_id = excluded.template_id,
                    intent_key = excluded.intent_key,
                    canonical_template = excluded.canonical_template,
                    search_text = excluded.search_text,
                    payload_json = excluded.payload_json,
                    schema_fingerprint = excluded.schema_fingerprint,
                    template_version = excluded.template_version,
                    status = excluded.status,
                    source = excluded.source,
                    success_count = excluded.success_count,
                    reuse_count = excluded.reuse_count,
                    shadow_match_count = excluded.shadow_match_count,
                    validation_failure_count = excluded.validation_failure_count,
                    vector_id = excluded.vector_id,
                    updated_at = excluded.updated_at,
                    last_used_at = excluded.last_used_at
                """,
                (
                    record["external_id"],
                    record["dataset_id"],
                    record["template_id"],
                    record["intent_key"],
                    record["canonical_template"],
                    record["search_text"],
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    record["schema_fingerprint"],
                    int(record["template_version"]),
                    record.get("status", "pending_review"),
                    record.get("source", "manual"),
                    int(record.get("success_count", (existing or {}).get("success_count", 0))),
                    int(record.get("reuse_count", (existing or {}).get("reuse_count", 0))),
                    int(record.get("shadow_match_count", (existing or {}).get("shadow_match_count", 0))),
                    int(record.get("validation_failure_count", (existing or {}).get("validation_failure_count", 0))),
                    vector_id,
                    now,
                    now,
                    record.get("last_used_at", (existing or {}).get("last_used_at")),
                ),
            )
        stored = self.get_sql_template(str(record["external_id"]))
        if stored is None:
            raise RuntimeError("SQL template upsert did not persist a record")
        return stored

    def get_sql_template(self, external_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM sql_templates WHERE external_id = ?",
            (external_id,),
        ).fetchone()
        return self._template_from_row(row) if row else None

    def find_sql_templates(
        self,
        dataset_id: str,
        intent_key: Optional[str] = None,
        canonical_template: Optional[str] = None,
        statuses: Sequence[str] = ("active",),
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conditions = ["dataset_id = ?"]
        params: List[Any] = [dataset_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if intent_key:
            conditions.append("intent_key = ?")
            params.append(intent_key)
        if canonical_template:
            conditions.append("canonical_template = ?")
            params.append(canonical_template)
        params.append(max(1, min(int(limit), 100)))
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM sql_templates
            WHERE {' AND '.join(conditions)}
            ORDER BY template_version DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._template_from_row(row) for row in rows]

    def list_sql_templates(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List SQL templates with lifecycle counters for administration."""
        conditions = ["1 = 1"]
        params: List[Any] = []
        if dataset_id:
            conditions.append("dataset_id = ?")
            params.append(dataset_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        params.append(max(1, min(int(limit), 500)))
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM sql_templates
            WHERE {' AND '.join(conditions)}
            ORDER BY reuse_count DESC, last_used_at DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._template_from_row(row) for row in rows]

    def get_sql_templates_by_vector_ids(
        self,
        vector_ids: Sequence[int],
        dataset_id: str,
        statuses: Sequence[str] = ("active",),
    ) -> Dict[int, Dict[str, Any]]:
        ids = [int(vector_id) for vector_id in vector_ids if int(vector_id) >= 0]
        if not ids:
            return {}
        id_placeholders = ",".join("?" for _ in ids)
        conditions = [
            f"vector_id IN ({id_placeholders})",
            "dataset_id = ?",
        ]
        params: List[Any] = [*ids, dataset_id]
        if statuses:
            status_placeholders = ",".join("?" for _ in statuses)
            conditions.append(f"status IN ({status_placeholders})")
            params.extend(statuses)
        rows = self.conn.execute(
            f"SELECT * FROM sql_templates WHERE {' AND '.join(conditions)}",
            params,
        ).fetchall()
        return {
            int(row["vector_id"]): self._template_from_row(row)
            for row in rows
            if row["vector_id"] is not None
        }

    def set_sql_template_status(
        self,
        external_id: str,
        status: str,
        vector_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE sql_templates
                SET status = ?, vector_id = ?, updated_at = ?
                WHERE external_id = ?
                """,
                (status, vector_id, now, external_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get_sql_template(external_id)

    def delete_sql_template(
        self,
        external_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Permanently delete one SQL template and return its previous record."""
        existing = self.get_sql_template(external_id)
        if existing is None:
            return None
        with self.conn:
            self.conn.execute(
                "DELETE FROM sql_templates WHERE external_id = ?",
                (external_id,),
            )
        return existing

    def record_sql_template_outcome(
        self,
        external_id: str,
        outcome: str,
    ) -> Optional[Dict[str, Any]]:
        counters = {
            "reuse": "reuse_count",
            "shadow_match": "shadow_match_count",
            "validation_failure": "validation_failure_count",
            "success": "success_count",
        }
        column = counters.get(outcome)
        if column is None:
            raise ValueError(f"Unsupported SQL template outcome: {outcome}")
        now = time.time()
        with self.conn:
            cursor = self.conn.execute(
                f"""
                UPDATE sql_templates
                SET {column} = {column} + 1,
                    updated_at = ?,
                    last_used_at = ?
                WHERE external_id = ?
                """,
                (now, now, external_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get_sql_template(external_id)

    def get_sql_template_metrics(self) -> Dict[str, Any]:
        status_rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM sql_templates
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()
        totals = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_templates,
                COALESCE(SUM(success_count), 0) AS success_count,
                COALESCE(SUM(reuse_count), 0) AS reuse_count,
                COALESCE(SUM(shadow_match_count), 0) AS shadow_match_count,
                COALESCE(SUM(validation_failure_count), 0)
                    AS validation_failure_count
            FROM sql_templates
            """
        ).fetchone()
        return {
            "total_templates": int(totals["total_templates"] or 0),
            "by_status": {
                str(row["status"]): int(row["count"])
                for row in status_rows
            },
            "success_count": int(totals["success_count"] or 0),
            "reuse_count": int(totals["reuse_count"] or 0),
            "shadow_match_count": int(totals["shadow_match_count"] or 0),
            "validation_failure_count": int(
                totals["validation_failure_count"] or 0
            ),
        }

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
            "sql_templates": self.get_sql_template_metrics(),
        }

    def checkpoint(self) -> None:
        """Flush the SQLite WAL before publishing a matching FAISS snapshot."""
        self.conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()

    def close(self) -> None:
        self.conn.close()
