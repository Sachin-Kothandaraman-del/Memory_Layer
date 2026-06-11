"""Supabase (Postgres + pgvector) memory store for cloud deployments.

Implements the same :class:`MemoryStore` interface as the SQLite backend, so
the whole memlayer pipeline (consolidation, forgetting curve, time travel,
audit) runs unchanged against a hosted database.

Requires the schema in ``supabase/schema.sql`` (memories + audit_log tables,
pgvector, and the RPC functions used for vector/keyword search and touch).
Uses the service-role key — run it ONLY on a server, never in a browser;
user scoping is enforced by the calling code via ``user_id`` filters.
"""

from __future__ import annotations

import json
import logging
import os
import time
import importlib
import importlib.machinery
import importlib.util
import sys
from typing import Any, Sequence

from ..models import AuditEntry, MemoryRecord, MemoryType
from .base import MemoryStore

logger = logging.getLogger("memlayer")


def _load_supabase_create_client():
    """Resolve supabase.create_client even if a local `supabase/` folder exists."""
    try:
        mod = importlib.import_module("supabase")
        create_client = getattr(mod, "create_client", None)
        if create_client is not None:
            return create_client
    except Exception:  # noqa: BLE001 - fallback below
        pass

    # Fallback: explicitly search non-project sys.path entries for the installed
    # package so namespace folders in the repo cannot shadow it.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for path in sys.path:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path.startswith(project_root):
            continue
        spec = importlib.machinery.PathFinder.find_spec("supabase", [path])
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        create_client = getattr(module, "create_client", None)
        if create_client is not None:
            return create_client

    raise ImportError(
        "Could not import `create_client` from installed `supabase` package. "
        "Ensure `supabase` is installed in the runtime environment."
    )


class SupabaseMemoryStore(MemoryStore):
    def __init__(
        self,
        client=None,
        url: str | None = None,
        service_role_key: str | None = None,
    ):
        if client is None:
            create_client = _load_supabase_create_client()
            url = url or os.environ["SUPABASE_URL"]
            service_role_key = service_role_key or os.environ[
                "SUPABASE_SERVICE_ROLE_KEY"
            ]
            client = create_client(url, service_role_key)
        self.client = client

    # -- CRUD ----------------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        self.client.table("memories").upsert(self._to_row(record)).execute()

    def get(self, memory_id: str) -> MemoryRecord | None:
        resp = (
            self.client.table("memories").select("*")
            .eq("id", memory_id).limit(1).execute()
        )
        return self._to_record(resp.data[0]) if resp.data else None

    def update(self, record: MemoryRecord) -> None:
        record.updated_at = time.time()
        self.add(record)

    def delete(self, memory_id: str) -> bool:
        resp = (
            self.client.table("memories").delete()
            .eq("id", memory_id).execute()
        )
        return bool(resp.data)

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
        resp = self.client.rpc("match_memories", {
            "query_embedding": list(embedding),
            "match_limit": limit,
            "p_user_id": user_id,
            "p_agent_id": agent_id,
            "p_session_id": session_id,
            "p_memory_type": memory_type.value if memory_type else None,
            "p_current_only": current_only,
            "p_as_of": as_of,
        }).execute()
        return [
            (self._to_record(row), float(row.get("similarity") or 0.0))
            for row in (resp.data or [])
        ]

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
        if not query.strip():
            return []
        resp = self.client.rpc("search_memories_text", {
            "search_query": query,
            "match_limit": limit,
            "p_user_id": user_id,
            "p_agent_id": agent_id,
            "p_session_id": session_id,
            "p_memory_type": memory_type.value if memory_type else None,
            "p_current_only": current_only,
            "p_as_of": as_of,
        }).execute()
        return [self._to_record(row) for row in (resp.data or [])]

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
        q = self.client.table("memories").select("*")
        q = self._apply_filters(q, user_id, agent_id, session_id, memory_type,
                                current_only, as_of)
        resp = q.order("updated_at", desc=True).limit(limit).execute()
        return [self._to_record(row) for row in (resp.data or [])]

    def touch(
        self,
        memory_ids: Sequence[str],
        strength_factor: float = 1.0,
        strength_max: float = 16.0,
    ) -> None:
        if not memory_ids:
            return
        self.client.rpc("touch_memories", {
            "p_ids": list(memory_ids),
            "p_factor": strength_factor,
            "p_max": strength_max,
        }).execute()

    def count(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        current_only: bool = True,
    ) -> int:
        q = self.client.table("memories").select("id", count="exact")
        q = self._apply_filters(q, user_id, None, None, memory_type,
                                current_only, None)
        resp = q.execute()
        return int(resp.count or 0)

    def users(self) -> list[str]:
        resp = self.client.rpc("distinct_users", {}).execute()
        return [row["user_id"] for row in (resp.data or [])]

    def predecessor(self, memory_id: str) -> MemoryRecord | None:
        resp = (
            self.client.table("memories").select("*")
            .eq("superseded_by", memory_id).limit(1).execute()
        )
        return self._to_record(resp.data[0]) if resp.data else None

    def clear(self, user_id: str | None = None) -> int:
        q = self.client.table("memories").delete()
        if user_id is None:
            q = q.neq("id", "")  # PostgREST requires a filter; match all
        else:
            q = q.eq("user_id", user_id)
        resp = q.execute()
        return len(resp.data or [])

    # -- Audit log ----------------------------------------------------------------

    def log_audit(self, entry: AuditEntry) -> None:
        try:
            self.client.table("audit_log").insert({
                "id": entry.id,
                "ts": entry.ts,
                "user_id": entry.user_id,
                "action": entry.action,
                "memory_id": entry.memory_id,
                "reasoning": entry.reasoning,
                "detail": entry.detail,
            }).execute()
        except Exception as exc:  # noqa: BLE001 - auditing must not break writes
            logger.error("supabase audit insert failed: %s", exc)

    def get_audit(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        q = self.client.table("audit_log").select("*")
        if memory_id is not None:
            q = q.eq("memory_id", memory_id)
        if user_id is not None:
            q = q.eq("user_id", user_id)
        resp = q.order("ts", desc=True).limit(limit).execute()
        return [
            AuditEntry(
                id=row["id"], ts=row["ts"], user_id=row["user_id"],
                action=row["action"], memory_id=row["memory_id"],
                reasoning=row["reasoning"], detail=row["detail"] or {},
            )
            for row in (resp.data or [])
        ]

    def close(self) -> None:
        pass  # HTTP client; nothing to close

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _apply_filters(q, user_id, agent_id, session_id, memory_type,
                       current_only, as_of):
        if user_id is not None:
            q = q.eq("user_id", user_id)
        if agent_id is not None:
            q = q.eq("agent_id", agent_id)
        if session_id is not None:
            q = q.eq("session_id", session_id)
        if memory_type is not None:
            q = q.eq("memory_type", memory_type.value)
        if as_of is not None:
            q = q.or_(f"valid_from.is.null,valid_from.lte.{as_of}")
            q = q.or_(f"valid_until.is.null,valid_until.gt.{as_of}")
        elif current_only:
            q = q.is_("valid_until", "null")
        return q

    @staticmethod
    def _to_row(record: MemoryRecord) -> dict[str, Any]:
        row = record.to_dict()
        row["embedding"] = (
            list(record.embedding) if record.embedding is not None else None
        )
        return row

    @staticmethod
    def _to_record(row: dict[str, Any]) -> MemoryRecord:
        embedding = row.get("embedding")
        if isinstance(embedding, str):  # PostgREST returns vectors as text
            embedding = json.loads(embedding)
        detail = dict(row)
        detail["embedding"] = embedding
        detail.pop("content_tsv", None)
        detail.pop("similarity", None)
        return MemoryRecord.from_dict(detail)
