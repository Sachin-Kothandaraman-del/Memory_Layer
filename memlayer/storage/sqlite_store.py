"""SQLite-backed memory store: vector search (numpy) + FTS5 keyword search,
fact validity windows (time travel), memory strength, and an audit log.

Zero external services — a single file on disk. Embeddings are stored as
float32 blobs; cosine similarity is computed in numpy over the filtered
candidate set, which is fast well into the tens of thousands of memories.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from typing import Sequence

import numpy as np

from ..models import AuditEntry, MemoryRecord, MemoryType
from .base import MemoryStore

logger = logging.getLogger("memlayer")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    memory_type      TEXT NOT NULL,
    content          TEXT NOT NULL,
    user_id          TEXT NOT NULL DEFAULT 'default',
    agent_id         TEXT,
    session_id       TEXT,
    importance       REAL NOT NULL DEFAULT 0.5,
    category         TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    source_ids       TEXT NOT NULL DEFAULT '[]',
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    strength         REAL NOT NULL DEFAULT 1.0,
    valid_from       REAL,
    valid_until      REAL,
    superseded_by    TEXT,
    embedding        BLOB,
    dim              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_mem_valid ON memories(valid_until);

CREATE TABLE IF NOT EXISTS audit_log (
    id        TEXT PRIMARY KEY,
    ts        REAL NOT NULL,
    user_id   TEXT,
    action    TEXT NOT NULL,
    memory_id TEXT,
    reasoning TEXT,
    detail    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_mem ON audit_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(id UNINDEXED, content);
"""

# columns added since v0.1 — applied to old databases on open
_MIGRATIONS = {
    "strength": "REAL NOT NULL DEFAULT 1.0",
    "valid_from": "REAL",
    "valid_until": "REAL",
    "superseded_by": "TEXT",
}


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = "memlayer.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._migrate()
        self._conn.executescript(_SCHEMA)
        self._fts_enabled = self._init_fts()

    def _migrate(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(memories)").fetchall()
        if not rows:
            return  # fresh database — full schema will be created
        existing = {r["name"] for r in rows}
        with self._conn:
            for col, decl in _MIGRATIONS.items():
                if col not in existing:
                    self._conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {col} {decl}"
                    )
            self._conn.execute(
                "UPDATE memories SET valid_from = created_at "
                "WHERE valid_from IS NULL"
            )

    def _init_fts(self) -> bool:
        try:
            self._conn.executescript(_FTS_SCHEMA)
            return True
        except sqlite3.OperationalError:  # pragma: no cover - FTS5 missing
            logger.warning("SQLite FTS5 unavailable; keyword search disabled")
            return False

    # -- CRUD ----------------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        blob, dim = self._pack_embedding(record.embedding)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, memory_type, content, user_id, agent_id, session_id,
                    importance, category, metadata, source_ids,
                    created_at, updated_at, last_accessed_at, access_count,
                    strength, valid_from, valid_until, superseded_by,
                    embedding, dim)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.id, record.memory_type.value, record.content,
                    record.user_id, record.agent_id, record.session_id,
                    record.importance, record.category,
                    json.dumps(record.metadata, ensure_ascii=False),
                    json.dumps(record.source_ids),
                    record.created_at, record.updated_at,
                    record.last_accessed_at, record.access_count,
                    record.strength, record.valid_from, record.valid_until,
                    record.superseded_by,
                    blob, dim,
                ),
            )
            if self._fts_enabled:
                self._conn.execute(
                    "DELETE FROM memories_fts WHERE id = ?", (record.id,)
                )
                self._conn.execute(
                    "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                    (record.id, record.content),
                )

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update(self, record: MemoryRecord) -> None:
        record.updated_at = time.time()
        self.add(record)  # INSERT OR REPLACE

    def delete(self, memory_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            if self._fts_enabled:
                self._conn.execute(
                    "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
                )
        return cur.rowcount > 0

    # -- Search ----------------------------------------------------------------

    def vector_search(
        self,
        embedding: Sequence[float],
        limit: int = 20,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        current_only: bool = True,
        as_of: float | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        where, params = self._filters(
            user_id, agent_id, session_id, memory_type, current_only, as_of
        )
        where.append("embedding IS NOT NULL")
        sql = f"SELECT * FROM memories WHERE {' AND '.join(where)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        matrix = np.stack(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        )
        # all vectors are stored normalized, so dot product == cosine similarity
        sims = matrix @ query
        order = np.argsort(-sims)[:limit]
        return [(self._row_to_record(rows[i]), float(sims[i])) for i in order]

    def keyword_search(
        self,
        query: str,
        limit: int = 20,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        current_only: bool = True,
        as_of: float | None = None,
    ) -> list[MemoryRecord]:
        if not self._fts_enabled:
            return []
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        where, params = self._filters(
            user_id, agent_id, session_id, memory_type, current_only, as_of
        )
        sql = (
            "SELECT m.* FROM memories_fts f "
            "JOIN memories m ON m.id = f.id "
            f"WHERE memories_fts MATCH ? AND {' AND '.join(where)} "
            "ORDER BY bm25(memories_fts) LIMIT ?"
        )
        with self._lock:
            try:
                rows = self._conn.execute(
                    sql, [match_expr, *params, limit]
                ).fetchall()
            except sqlite3.OperationalError:
                return []  # unparseable FTS query — treat as no keyword hits
        return [self._row_to_record(r) for r in rows]

    def list(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 1000,
        current_only: bool = True,
        as_of: float | None = None,
    ) -> list[MemoryRecord]:
        where, params = self._filters(
            user_id, agent_id, session_id, memory_type, current_only, as_of
        )
        sql = (
            f"SELECT * FROM memories WHERE {' AND '.join(where)} "
            "ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, [*params, limit]).fetchall()
        return [self._row_to_record(r) for r in rows]

    def touch(
        self,
        memory_ids: Sequence[str],
        strength_factor: float = 1.0,
        strength_max: float = 16.0,
    ) -> None:
        if not memory_ids:
            return
        now = time.time()
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock, self._conn:
            self._conn.execute(
                f"""UPDATE memories
                    SET last_accessed_at = ?,
                        access_count = access_count + 1,
                        strength = MIN(strength * ?, ?)
                    WHERE id IN ({placeholders})""",
                [now, strength_factor, strength_max, *memory_ids],
            )

    def count(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        current_only: bool = True,
    ) -> int:
        where, params = self._filters(
            user_id, None, None, memory_type, current_only, None
        )
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
        return int(row["n"])

    def users(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM memories ORDER BY user_id"
            ).fetchall()
        return [r["user_id"] for r in rows]

    def predecessor(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE superseded_by = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear(self, user_id: str | None = None) -> int:
        with self._lock, self._conn:
            if user_id is None:
                cur = self._conn.execute("DELETE FROM memories")
                if self._fts_enabled:
                    self._conn.execute("DELETE FROM memories_fts")
            else:
                ids = [
                    r["id"]
                    for r in self._conn.execute(
                        "SELECT id FROM memories WHERE user_id = ?", (user_id,)
                    )
                ]
                cur = self._conn.execute(
                    "DELETE FROM memories WHERE user_id = ?", (user_id,)
                )
                if self._fts_enabled and ids:
                    placeholders = ",".join("?" * len(ids))
                    self._conn.execute(
                        f"DELETE FROM memories_fts WHERE id IN ({placeholders})",
                        ids,
                    )
        return cur.rowcount

    # -- Audit log ----------------------------------------------------------------

    def log_audit(self, entry: AuditEntry) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO audit_log
                   (id, ts, user_id, action, memory_id, reasoning, detail)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    entry.id, entry.ts, entry.user_id, entry.action,
                    entry.memory_id, entry.reasoning,
                    json.dumps(entry.detail, ensure_ascii=False),
                ),
            )

    def get_audit(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        where, params = ["1=1"], []
        if memory_id is not None:
            where.append("memory_id = ?")
            params.append(memory_id)
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_log WHERE {' AND '.join(where)} "
                "ORDER BY ts DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [
            AuditEntry(
                id=r["id"], ts=r["ts"], user_id=r["user_id"],
                action=r["action"], memory_id=r["memory_id"],
                reasoning=r["reasoning"], detail=json.loads(r["detail"]),
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _filters(
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        memory_type: MemoryType | None,
        current_only: bool = True,
        as_of: float | None = None,
    ) -> tuple[list[str], list]:
        where, params = ["1=1"], []
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if memory_type is not None:
            where.append("memory_type = ?")
            params.append(memory_type.value)
        if as_of is not None:
            where.append("(valid_from IS NULL OR valid_from <= ?)")
            params.append(as_of)
            where.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(as_of)
        elif current_only:
            where.append("valid_until IS NULL")
        return where, params

    @staticmethod
    def _pack_embedding(
        embedding: Sequence[float] | None,
    ) -> tuple[bytes | None, int | None]:
        if embedding is None:
            return None, None
        arr = np.asarray(embedding, dtype=np.float32)
        return arr.tobytes(), int(arr.shape[0])

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        embedding = None
        if row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
        return MemoryRecord(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            importance=row["importance"],
            category=row["category"],
            metadata=json.loads(row["metadata"]),
            source_ids=json.loads(row["source_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            strength=row["strength"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            superseded_by=row["superseded_by"],
            embedding=embedding,
        )
