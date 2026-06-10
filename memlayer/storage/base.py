"""Abstract storage interface — implement this to plug in another backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import MemoryRecord, MemoryType


class MemoryStore(ABC):
    """Persistence + search backend for memory records.

    All filter parameters follow the same convention: ``None`` means
    "don't filter on this field".
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
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    def touch(self, memory_ids: Sequence[str]) -> None:
        """Mark memories as accessed (reinforcement: bumps access stats)."""

    @abstractmethod
    def count(self, user_id: str | None = None,
              memory_type: MemoryType | None = None) -> int: ...

    def users(self) -> list[str]:
        """Distinct user ids present in the store (optional capability)."""
        return []

    @abstractmethod
    def clear(self, user_id: str | None = None) -> int:
        """Delete memories (all, or for one user). Returns count deleted."""

    @abstractmethod
    def close(self) -> None: ...
