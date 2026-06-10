from __future__ import annotations

import time

from memlayer import MemoryConfig
from memlayer.models import MemoryRecord, MemoryType
from memlayer.retrieval import Retriever
from memlayer.storage.sqlite_store import SQLiteMemoryStore


def build(store, embedder, **config_overrides) -> Retriever:
    config = MemoryConfig(embed_dim=embedder.dim, **config_overrides)
    return Retriever(store, embedder, config)


def seed(store, embedder, content: str, **kwargs) -> MemoryRecord:
    rec = MemoryRecord(
        content=content,
        embedding=embedder.embed_documents([content])[0],
        **kwargs,
    )
    store.add(rec)
    return rec


def test_relevant_memory_ranks_first(store, fake_embedder):
    seed(store, fake_embedder, "user loves italian food and pasta carbonara")
    seed(store, fake_embedder, "user works as a backend engineer at acme")
    seed(store, fake_embedder, "user has a golden retriever named biscuit")
    retriever = build(store, fake_embedder)

    results = retriever.search("what food does the user like to eat pasta")
    assert results, "expected results"
    assert "italian food" in results[0].record.content


def test_keyword_only_match_is_recoverable(store, fake_embedder):
    # a term present verbatim should surface via FTS even if cosine is weak
    seed(store, fake_embedder, "deployment runbook stored in confluence XYZZY-42")
    seed(store, fake_embedder, "user prefers short answers")
    retriever = build(store, fake_embedder)
    results = retriever.search("XYZZY-42")
    assert any("XYZZY-42" in s.record.content for s in results)


def test_recency_breaks_ties(store, fake_embedder):
    old = seed(store, fake_embedder, "user timezone is UTC plus one")
    new = seed(store, fake_embedder, "user timezone is UTC plus one again")
    # backdate the old record by 60 days
    old.created_at = old.updated_at = time.time() - 60 * 86400
    store.add(old)

    retriever = build(store, fake_embedder, weight_recency=0.3,
                      weight_similarity=0.5, weight_importance=0.2)
    results = retriever.search("user timezone", limit=2)
    by_id = {s.record.id: s for s in results}
    assert by_id[new.id].recency > by_id[old.id].recency
    assert by_id[new.id].score > by_id[old.id].score


def test_importance_contributes_to_score(store, fake_embedder):
    low = seed(store, fake_embedder, "user mentioned liking coffee", importance=0.1)
    high = seed(store, fake_embedder, "user mentioned liking coffee a lot",
                importance=0.95)
    retriever = build(store, fake_embedder)
    results = retriever.search("coffee", limit=2, reinforce=False)
    by_id = {s.record.id: s for s in results}
    assert by_id[high.id].importance > by_id[low.id].importance


def test_mmr_prefers_diverse_results(store, fake_embedder):
    seed(store, fake_embedder, "user likes pizza pizza pizza")
    seed(store, fake_embedder, "user likes pizza pizza pizza so much")
    seed(store, fake_embedder, "user deadline for the report is friday")
    retriever = build(store, fake_embedder, mmr_lambda=0.3)
    results = retriever.search("user likes pizza deadline", limit=2)
    contents = " | ".join(s.record.content for s in results)
    assert "deadline" in contents  # diversity pulled in the non-duplicate


def test_reinforcement_touches_results(store, fake_embedder):
    rec = seed(store, fake_embedder, "user studied physics in college")
    retriever = build(store, fake_embedder)
    retriever.search("physics college")
    assert store.get(rec.id).access_count == 1
    retriever.search("physics college", reinforce=False)
    assert store.get(rec.id).access_count == 1


def test_type_filter(store, fake_embedder):
    seed(store, fake_embedder, "episodic note about pandas",
         memory_type=MemoryType.EPISODIC)
    seed(store, fake_embedder, "semantic fact about pandas",
         memory_type=MemoryType.SEMANTIC)
    retriever = build(store, fake_embedder)
    results = retriever.search("pandas", memory_type=MemoryType.SEMANTIC)
    assert results
    assert all(s.record.memory_type == MemoryType.SEMANTIC for s in results)
