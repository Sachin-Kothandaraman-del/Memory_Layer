"""Core data models for episodic and semantic memories."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"   # raw events: what was said / what happened, when
    SEMANTIC = "semantic"   # distilled, durable facts extracted from episodes


class FactCategory(str, Enum):
    IDENTITY = "identity"          # who the user is (name, role, location...)
    PREFERENCE = "preference"      # likes, dislikes, style choices
    GOAL = "goal"                  # what the user is trying to achieve
    RELATIONSHIP = "relationship"  # people / orgs the user relates to
    CONSTRAINT = "constraint"      # hard rules ("never email after 6pm")
    EVENT = "event"                # notable dated occurrences
    KNOWLEDGE = "knowledge"        # domain facts the agent should retain
    INSIGHT = "insight"            # higher-order pattern found by reflection
    OTHER = "other"


def now_ts() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryRecord:
    """A single stored memory (episodic or semantic)."""

    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    id: str = field(default_factory=new_id)
    user_id: str = "default"
    agent_id: str | None = None
    session_id: str | None = None
    importance: float = 0.5            # 0..1, contributes to retrieval score
    category: str | None = None        # FactCategory value for semantic memories
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)  # provenance links
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    last_accessed_at: float = field(default_factory=now_ts)
    access_count: int = 0
    strength: float = 1.0              # forgetting curve: grows with each recall
    valid_from: float | None = None    # time-travel: when this version became true
    valid_until: float | None = None   # None = still current
    superseded_by: str | None = None   # id of the version that replaced this one
    embedding: list[float] | None = None  # populated on demand, not in to_dict()

    def __post_init__(self) -> None:
        if self.valid_from is None:
            self.valid_from = self.created_at

    @property
    def is_current(self) -> bool:
        return self.valid_until is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "importance": self.importance,
            "category": self.category,
            "metadata": self.metadata,
            "source_ids": self.source_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "strength": self.strength,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=d["id"],
            memory_type=MemoryType(d["memory_type"]),
            content=d["content"],
            user_id=d.get("user_id", "default"),
            agent_id=d.get("agent_id"),
            session_id=d.get("session_id"),
            importance=d.get("importance", 0.5),
            category=d.get("category"),
            metadata=d.get("metadata") or {},
            source_ids=d.get("source_ids") or [],
            created_at=d.get("created_at", now_ts()),
            updated_at=d.get("updated_at", now_ts()),
            last_accessed_at=d.get("last_accessed_at", now_ts()),
            access_count=d.get("access_count", 0),
            strength=d.get("strength", 1.0),
            valid_from=d.get("valid_from"),
            valid_until=d.get("valid_until"),
            superseded_by=d.get("superseded_by"),
            embedding=d.get("embedding"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ScoredMemory:
    """A retrieval result: the record plus its score breakdown."""

    record: MemoryRecord
    similarity: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    score: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ScoredMemory(score={self.score:.3f} sim={self.similarity:.3f} "
            f"rec={self.recency:.3f} imp={self.importance:.3f} "
            f"[{self.record.memory_type.value}] {self.record.content[:60]!r})"
        )


@dataclass
class ExtractedFact:
    """A candidate semantic fact produced by the extraction LLM."""

    content: str
    category: str = FactCategory.OTHER.value
    importance: float = 0.6


@dataclass
class AuditEntry:
    """One entry in the glass-box audit log: what changed in memory and why."""

    action: str                     # ADD/UPDATE/RETRACT/NONE/FORGET/CLEAR/
                                    # PRUNE/REFLECT/REDACT/SKIPPED_PRIVATE
    user_id: str | None = None
    memory_id: str | None = None
    reasoning: str | None = None    # the LLM's stated reasoning, when available
    detail: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    ts: float = field(default_factory=now_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "user_id": self.user_id,
            "action": self.action,
            "memory_id": self.memory_id,
            "reasoning": self.reasoning,
            "detail": self.detail,
        }
