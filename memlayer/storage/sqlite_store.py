"""SQLite-backed memory store: vector search (numpy) + FTS5 keyword search.

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

from ..models import MemoryRecord, MemoryType
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
    embedding        BLOB,
    dim              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(id UNINDEXED, content);
"""


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = "memlayer.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._fts_enabled = self._init_fts()

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
                    embedding, dim)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.id, record.memory_type.value, record.content,
                    record.user_id, record.agent_id, record.session_id,
                    record.importance, record.category,
                    json.dumps(record.metadata, ensure_ascii=False),
                    json.dumps(record.source_ids),
                    record.created_at, record.updated_at,
                    record.last_accessed_at, record.access_count,
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
    ) -> list[tuple[MemoryRecord, float]]:
        where, params = self._filters(user_id, agent_id, session_id, memory_type)
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
    ) -> list[MemoryRecord]:
        if not self._fts_enabled:
            return []
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        where, params = self._filters(user_id, agent_id, session_id, memory_type)
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
    ) -> list[MemoryRecord]:
        where, params = self._filters(user_id, agent_id, session_id, memory_type)
        sql = (
            f"SELECT * FROM memories WHERE {' AND '.join(where)} "
            "ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, [*params, limit]).fetchall()
        return [self._row_to_record(r) for r in rows]

    def touch(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        now = time.time()
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock, self._conn:
            self._conn.execute(
                f"""UPDATE memories
                    SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE id IN ({placeholders})""",
                [now, *memory_ids],
            )

    def count(self, user_id: str | None = None,
              memory_type: MemoryType | None = None) -> int:
        where, params = self._filters(user_id, None, None, memory_type)
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
            embedding=embedding,
        )
