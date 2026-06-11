"""Offline tests for SupabaseMemoryStore's row/record mapping.

The live store needs a real Supabase project (see DEPLOY.md); these tests
cover the serialization logic that doesn't require one.
"""

from __future__ import annotations

from memlayer.models import MemoryRecord, MemoryType
from memlayer.storage.supabase_store import SupabaseMemoryStore


def test_row_roundtrip_preserves_everything():
    rec = MemoryRecord(
        content="User lives in Munich",
        memory_type=MemoryType.SEMANTIC,
        user_id="auth-user-123",
        importance=0.9,
        category="identity",
        metadata={"k": "v"},
        source_ids=["ep1", "ep2"],
        strength=2.4,
        valid_until=1750000000.0,
        superseded_by="newer-id",
        embedding=[0.25, -0.5, 0.75],
    )
    row = SupabaseMemoryStore._to_row(rec)
    assert row["embedding"] == [0.25, -0.5, 0.75]
    assert row["metadata"] == {"k": "v"}

    back = SupabaseMemoryStore._to_record(row)
    assert back.id == rec.id
    assert back.content == rec.content
    assert back.memory_type == MemoryType.SEMANTIC
    assert back.user_id == "auth-user-123"
    assert back.source_ids == ["ep1", "ep2"]
    assert back.strength == 2.4
    assert back.valid_until == 1750000000.0
    assert back.superseded_by == "newer-id"
    assert back.embedding == [0.25, -0.5, 0.75]


def test_record_parses_postgrest_vector_string():
    # PostgREST returns pgvector columns as a text literal
    row = MemoryRecord(content="x", embedding=None).to_dict()
    row["embedding"] = "[0.1, 0.2, 0.3]"
    row["content_tsv"] = "'x':1"      # generated column comes back too
    row["similarity"] = 0.87          # RPC result includes the score
    rec = SupabaseMemoryStore._to_record(row)
    assert rec.embedding == [0.1, 0.2, 0.3]


def test_row_without_embedding():
    rec = MemoryRecord(content="no vector yet")
    row = SupabaseMemoryStore._to_row(rec)
    assert row["embedding"] is None
    assert SupabaseMemoryStore._to_record(row).embedding is None
