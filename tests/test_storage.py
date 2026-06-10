from __future__ import annotations

import numpy as np

from memlayer.models import MemoryRecord, MemoryType
from memlayer.storage.sqlite_store import SQLiteMemoryStore


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def make_record(content: str, embedding=None, **kwargs) -> MemoryRecord:
    return MemoryRecord(content=content, embedding=embedding, **kwargs)


def test_add_get_roundtrip(store):
    rec = make_record(
        "User prefers tabs over spaces",
        embedding=_unit([1, 0, 0]),
        memory_type=MemoryType.SEMANTIC,
        user_id="u1",
        category="preference",
        metadata={"foo": "bar"},
        source_ids=["ep1"],
    )
    store.add(rec)
    loaded = store.get(rec.id)
    assert loaded is not None
    assert loaded.content == rec.content
    assert loaded.memory_type == MemoryType.SEMANTIC
    assert loaded.metadata == {"foo": "bar"}
    assert loaded.source_ids == ["ep1"]
    assert np.allclose(loaded.embedding, rec.embedding, atol=1e-6)


def test_update_and_delete(store):
    rec = make_record("original", embedding=_unit([1, 1, 0]))
    store.add(rec)
    rec.content = "revised"
    store.update(rec)
    assert store.get(rec.id).content == "revised"
    assert store.delete(rec.id) is True
    assert store.get(rec.id) is None
    assert store.delete(rec.id) is False


def test_vector_search_orders_by_similarity(store):
    target = _unit([1, 0, 0])
    close = make_record("close", embedding=_unit([0.95, 0.05, 0]))
    far = make_record("far", embedding=_unit([0, 1, 0]))
    mid = make_record("mid", embedding=_unit([0.5, 0.5, 0]))
    for r in (far, mid, close):
        store.add(r)
    results = store.vector_search(target, limit=3)
    assert [r.content for r, _ in results] == ["close", "mid", "far"]
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)


def test_search_respects_namespace_filters(store):
    a = make_record("alpha fact", embedding=_unit([1, 0, 0]), user_id="alice")
    b = make_record("beta fact", embedding=_unit([1, 0, 0]), user_id="bob")
    store.add(a)
    store.add(b)
    results = store.vector_search(_unit([1, 0, 0]), user_id="alice")
    assert [r.user_id for r, _ in results] == ["alice"]
    kw = store.keyword_search("fact", user_id="bob")
    assert [r.user_id for r in kw] == ["bob"]


def test_keyword_search_finds_terms(store):
    store.add(make_record("the quarterly revenue report is due friday"))
    store.add(make_record("user enjoys rock climbing on weekends"))
    hits = store.keyword_search("revenue report")
    assert len(hits) == 1
    assert "revenue" in hits[0].content
    assert store.keyword_search("!!! ???") == []  # symbols-only query is safe


def test_keyword_search_updated_after_delete(store):
    rec = make_record("findable unique zanzibar token")
    store.add(rec)
    assert len(store.keyword_search("zanzibar")) == 1
    store.delete(rec.id)
    assert store.keyword_search("zanzibar") == []


def test_touch_bumps_access_stats(store):
    rec = make_record("touched memory")
    store.add(rec)
    before = store.get(rec.id)
    store.touch([rec.id])
    after = store.get(rec.id)
    assert after.access_count == before.access_count + 1
    assert after.last_accessed_at >= before.last_accessed_at


def test_count_and_clear_scoped_by_user(store):
    store.add(make_record("a", user_id="u1"))
    store.add(make_record("b", user_id="u1", memory_type=MemoryType.SEMANTIC))
    store.add(make_record("c", user_id="u2"))
    assert store.count() == 3
    assert store.count(user_id="u1") == 2
    assert store.count(user_id="u1", memory_type=MemoryType.SEMANTIC) == 1
    assert store.clear(user_id="u1") == 2
    assert store.count() == 1
    assert store.clear() == 1
    assert store.count() == 0


def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "mem.db")
    s1 = SQLiteMemoryStore(db)
    rec = make_record("durable", embedding=_unit([1, 2, 3]))
    s1.add(rec)
    s1.close()
    s2 = SQLiteMemoryStore(db)
    loaded = s2.get(rec.id)
    assert loaded is not None and loaded.content == "durable"
    assert len(s2.keyword_search("durable")) == 1
    s2.close()
