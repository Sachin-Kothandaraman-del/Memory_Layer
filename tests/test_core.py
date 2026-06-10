from __future__ import annotations

import json

from memlayer.models import MemoryType


def test_add_stores_episodic_and_extracted_facts(memory, fake_llm):
    fake_llm.json_queue.append(
        {
            "facts": [
                {"content": "User's name is Priya", "category": "identity",
                 "importance": 0.9},
                {"content": "User leads the data platform team",
                 "category": "identity", "importance": 0.8},
            ]
        }
    )
    result = memory.add(
        "user: Hi, I'm Priya — I lead the data platform team at Acme.",
        user_id="u1",
    )
    assert result["episodic"] is not None
    assert len(result["facts"]) == 2
    stats = memory.stats(user_id="u1")
    assert stats["episodic"] == 1
    assert stats["semantic"] == 2


def test_add_skips_blank_content(memory):
    result = memory.add("   ")
    assert result == {"episodic": None, "facts": []}
    assert memory.stats()["total"] == 0


def test_infer_false_skips_extraction(memory, fake_llm):
    memory.add("user: remember this verbatim", user_id="u1", infer=False)
    assert memory.stats(user_id="u1")["semantic"] == 0
    assert all(c["kind"] != "json" for c in fake_llm.calls)


def test_consolidation_update_path(memory, fake_llm):
    # first write: one fact, no similar memories -> direct ADD (no LLM decide)
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Berlin", "category": "identity",
                    "importance": 0.9}]}
    )
    memory.add("user: I live in Berlin", user_id="u1")
    existing = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)
    assert len(existing) == 1
    old_id = existing[0].id

    # second write: near-duplicate fact -> LLM chooses UPDATE
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Munich", "category": "identity",
                    "importance": 0.9}]}
    )
    fake_llm.json_queue.append(
        {"operations": [{"op": "UPDATE", "id": old_id,
                         "content": "User lives in Munich (moved from Berlin)"}]}
    )
    memory.add("user: actually I moved to Munich", user_id="u1")

    facts = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)
    assert len(facts) == 1
    assert "Munich" in facts[0].content
    assert facts[0].id == old_id


def test_consolidation_delete_and_add(memory, fake_llm):
    # ensure the near-duplicate is "close enough" for the fake bag-of-words
    # embedder so the LLM decision path (not the fast-path ADD) is exercised
    memory.config.consolidation_sim_threshold = 0.5
    fake_llm.json_queue.append(
        {"facts": [{"content": "User is vegetarian", "category": "preference",
                    "importance": 0.8}]}
    )
    memory.add("user: I'm vegetarian", user_id="u1")
    old = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)[0]

    fake_llm.json_queue.append(
        {"facts": [{"content": "User is vegetarian no longer and eats fish",
                    "category": "preference", "importance": 0.8}]}
    )
    fake_llm.json_queue.append(
        {"operations": [{"op": "DELETE", "id": old.id}, {"op": "ADD"}]}
    )
    memory.add("user: I eat fish now", user_id="u1")

    facts = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)
    assert len(facts) == 1
    assert "fish" in facts[0].content


def test_get_context_formats_and_respects_budget(memory, fake_llm):
    fake_llm.json_queue.append(
        {"facts": [{"content": "User's favorite language is Python",
                    "category": "preference", "importance": 0.8}]}
    )
    memory.add("user: I mostly write Python these days", user_id="u1")

    context = memory.get_context("python language", user_id="u1")
    assert "Long-term memory" in context
    assert "favorite language is Python" in context
    assert "Known facts:" in context

    tiny = memory.get_context("python language", user_id="u1", token_budget=18)
    assert len(tiny) < len(context)

    assert memory.get_context("anything", user_id="nobody") == ""


def test_background_write_and_flush(memory):
    future = memory.add("user: async note about kubernetes", user_id="u1",
                        infer=False, wait=False)
    memory.flush()
    assert future.done()
    assert memory.stats(user_id="u1")["episodic"] == 1


def test_forget_and_clear(memory):
    res = memory.add("user: note one", user_id="u1", infer=False)
    memory.add("user: note two", user_id="u1", infer=False)
    assert memory.forget(res["episodic"]) is True
    assert memory.stats(user_id="u1")["total"] == 1
    assert memory.clear(user_id="u1") == 1
    assert memory.stats(user_id="u1")["total"] == 0


def test_prune_removes_stale_low_value(memory):
    import time as _t

    res = memory.add("user: ancient trivial note", user_id="u1", infer=False)
    rec = memory.get(res["episodic"])
    rec.created_at = rec.updated_at = _t.time() - 365 * 86400
    memory.store.add(rec)
    memory.add("user: fresh note", user_id="u1", infer=False)

    deleted = memory.prune(max_age_days=90, user_id="u1")
    assert deleted == 1
    assert memory.stats(user_id="u1")["episodic"] == 1


def test_export_import_roundtrip(memory):
    memory.add("user: exportable note", user_id="u1", infer=False)
    payload = memory.export(user_id="u1")
    records = json.loads(payload)
    assert len(records) == 1

    memory.clear()
    assert memory.stats()["total"] == 0
    count = memory.import_json(payload)
    assert count == 1
    hits = memory.search("exportable note", user_id="u1")
    assert hits and "exportable" in hits[0].record.content


def test_summarize_session(memory, fake_llm):
    memory.add("user: we decided to use postgres", user_id="u1",
               session_id="s1", infer=False)
    memory.add("user: also agreed on a march deadline", user_id="u1",
               session_id="s1", infer=False)
    fake_llm.text_queue.append("Team chose Postgres with a March deadline.")
    summary = memory.summarize_session("s1", user_id="u1")
    assert summary is not None
    assert summary.memory_type == MemoryType.SEMANTIC
    assert "Postgres" in summary.content
    assert len(summary.source_ids) == 2
