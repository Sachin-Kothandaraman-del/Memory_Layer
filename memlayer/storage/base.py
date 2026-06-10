"""Abstract storage interface — implement this to plug in another backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import AuditEntry, MemoryRecord, MemoryType


class MemoryStore(ABC):
    """Persistence + search backend for memory records.

    Filter conventions: ``None`` means "don't filter on this field".
    Validity: by default only *current* memories (``valid_until IS NULL``)
    are returned; pass ``current_only=False`` for full history, or ``as_of``
    (epoch seconds) to query what was believed at a moment in time.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryRecord | None: ...

    @abstractmethod
    def update(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
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
        """Return (record, cosine_similarity) pairs, best first."""

    @abstractmethod
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
        """Full-text search, best first. May return [] if unsupported."""

    @abstractmethod
    def list(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 1000,
        current_only: bool = True,
        as_of: float | None = None,
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    def touch(
        self,
        memory_ids: Sequence[str],
        strength_factor: float = 1.0,
        strength_max: float = 16.0,
    ) -> None:
        """Reinforce memories: bump access stats and multiply strength."""

    @abstractmethod
    def count(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        current_only: bool = True,
    ) -> int: ...

    @abstractmethod
    def clear(self, user_id: str | None = None) -> int:
        """Delete memories (all, or for one user). Returns count deleted."""

    @abstractmethod
    def close(self) -> None: ...

    # -- optional capabilities (override where supported) ----------------------

    def users(self) -> list[str]:
        """Distinct user ids present in the store."""
        return []

    def predecessor(self, memory_id: str) -> MemoryRecord | None:
        """The version that was superseded by ``memory_id``, if any."""
        return None

    def log_audit(self, entry: AuditEntry) -> None:
        """Record a glass-box audit entry (no-op if unsupported)."""

    def get_audit(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        return []
