"""Tests for the differentiating features: forgetting curve, time travel,
glass-box audit/provenance, reflection, and privacy guards."""

from __future__ import annotations

import time

from memlayer.models import MemoryRecord, MemoryType
from memlayer.privacy import redact_pii
from memlayer.retrieval import retention_of

DAY = 86400.0


def _backdate(memory, memory_id: str, days: float) -> None:
    rec = memory.get(memory_id)
    rec.created_at = rec.updated_at = time.time() - days * DAY
    rec.last_accessed_at = rec.updated_at
    memory.store.add(rec)


# ----------------------------------------------------------- forgetting curve

def test_faded_memory_excluded_but_recoverable(memory):
    res = memory.add("user: obscure trivia about xenon lighting", user_id="u1",
                     infer=False)
    _backdate(memory, res["episodic"], days=400)

    hits = memory.search("xenon lighting trivia", user_id="u1")
    assert all(h.record.id != res["episodic"] for h in hits), "should have faded"

    hits = memory.search("xenon lighting trivia", user_id="u1",
                         include_faded=True)
    assert any(h.record.id == res["episodic"] for h in hits), "faded != deleted"
    assert memory.stats(user_id="u1")["episodic"] == 1  # still stored


def test_recall_reinforces_strength(memory):
    res = memory.add("user: enjoys gardening tomatoes on sundays", user_id="u1",
                     infer=False)
    assert memory.get(res["episodic"]).strength == 1.0
    memory.search("gardening tomatoes", user_id="u1")
    reinforced = memory.get(res["episodic"]).strength
    assert reinforced > 1.0
    memory.search("gardening tomatoes", user_id="u1")
    assert memory.get(res["episodic"]).strength > reinforced


def test_stronger_memories_decay_slower(memory):
    now = time.time()
    weak = MemoryRecord(content="x", strength=1.0)
    strong = MemoryRecord(content="x", strength=8.0)
    for r in (weak, strong):
        r.updated_at = r.last_accessed_at = now - 14 * DAY
    assert retention_of(strong, memory.config, now) > retention_of(
        weak, memory.config, now
    )


def test_prune_by_retention(memory):
    res = memory.add("user: stale unrecalled note", user_id="u1", infer=False)
    _backdate(memory, res["episodic"], days=400)
    memory.add("user: fresh note", user_id="u1", infer=False)
    deleted = memory.prune(max_age_days=10_000, min_retention=0.05,
                           user_id="u1")
    assert deleted == 1
    assert memory.stats(user_id="u1")["episodic"] == 1


# --------------------------------------------------------------- time travel

def _move_cities(memory, fake_llm) -> tuple[str, str, float]:
    """Berlin -> Munich supersession. Returns (old_id, new_id, t_between)."""
    memory.config.consolidation_sim_threshold = 0.5
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Berlin", "category": "identity",
                    "importance": 0.9}]}
    )
    memory.add("user: I live in Berlin", user_id="u1")
    old = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)[0]
    time.sleep(0.02)
    t_between = time.time()
    time.sleep(0.02)
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Munich", "category": "identity",
                    "importance": 0.9}]}
    )
    fake_llm.json_queue.append(
        {"operations": [{"op": "UPDATE", "id": old.id,
                         "content": "User lives in Munich",
                         "reasoning": "user moved from Berlin to Munich"}]}
    )
    memory.add("user: I have moved to Munich now", user_id="u1")
    new = memory.store.list(user_id="u1", memory_type=MemoryType.SEMANTIC)[0]
    return old.id, new.id, t_between


def test_as_of_returns_past_beliefs(memory, fake_llm):
    old_id, new_id, t_between = _move_cities(memory, fake_llm)

    # today: Munich is the truth
    now_hits = memory.search("which city does the user live in", user_id="u1",
                             memory_type=MemoryType.SEMANTIC)
    assert any(h.record.id == new_id for h in now_hits)
    assert all(h.record.id != old_id for h in now_hits)

    # as of t_between: Berlin was the truth
    past_hits = memory.search("which city does the user live in", user_id="u1",
                              memory_type=MemoryType.SEMANTIC, as_of=t_between)
    assert any(h.record.id == old_id for h in past_hits)
    assert all(h.record.id != new_id for h in past_hits)


def test_history_shows_belief_timeline(memory, fake_llm):
    old_id, new_id, _ = _move_cities(memory, fake_llm)
    h = memory.history(new_id)
    assert [v["id"] for v in h["versions"]] == [old_id, new_id]
    assert "Berlin" in h["versions"][0]["content"]
    assert h["versions"][0]["valid_until"] is not None
    assert h["versions"][1]["valid_until"] is None
    assert h["sources"], "fact should cite its source episodes"
    # walking from the OLD record finds the same chain
    h_old = memory.history(old_id)
    assert [v["id"] for v in h_old["versions"]] == [old_id, new_id]


# ----------------------------------------------------------------- glass box

def test_audit_trail_with_reasoning(memory, fake_llm):
    old_id, new_id, _ = _move_cities(memory, fake_llm)
    entries = memory.audit_log(user_id="u1")
    actions = [e["action"] for e in entries]
    assert "ADD" in actions and "UPDATE" in actions
    update = next(e for e in entries if e["action"] == "UPDATE")
    assert update["memory_id"] == new_id
    assert update["reasoning"] == "user moved from Berlin to Munich"
    assert update["detail"]["old_id"] == old_id


def test_forget_and_clear_are_audited(memory):
    res = memory.add("user: temporary note", user_id="u1", infer=False)
    memory.forget(res["episodic"])
    memory.add("user: another note", user_id="u1", infer=False)
    memory.clear(user_id="u1")
    actions = [e["action"] for e in memory.audit_log(user_id="u1")]
    assert "FORGET" in actions and "CLEAR" in actions


# ----------------------------------------------------------------- reflection

def test_reflection_creates_evidence_backed_insight(memory, fake_llm):
    r1 = memory.add("user: shipped the deploy late again, sorry team",
                    user_id="u1", infer=False)
    r2 = memory.add("user: missed standup, was firefighting the deploy",
                    user_id="u1", infer=False)
    evidence = [r1["episodic"], r2["episodic"]]
    fake_llm.json_queue.append(
        {"insights": [{
            "insight": "User is under recurring deployment pressure",
            "evidence": evidence,
            "confidence": 0.8,
            "reasoning": "two separate episodes mention deploy trouble",
        }]}
    )
    out = memory.reflect(user_id="u1")
    assert out["examined_episodes"] == 2
    assert len(out["insights"]) == 1

    rec = memory.get(out["insights"][0]["id"])
    assert rec.category == "insight"
    assert rec.memory_type == MemoryType.SEMANTIC
    assert set(rec.source_ids) == set(evidence)
    assert any(e["action"] == "REFLECT" for e in memory.audit_log(user_id="u1"))


def test_reflection_ignores_hallucinated_evidence(memory, fake_llm):
    memory.add("user: one single episode", user_id="u1", infer=False)
    fake_llm.json_queue.append(
        {"insights": [{"insight": "Something", "evidence": ["not-a-real-id"],
                       "confidence": 0.9}]}
    )
    out = memory.reflect(user_id="u1")
    rec = memory.get(out["insights"][0]["id"])
    assert rec.source_ids == []  # invented evidence ids are dropped


def test_reflection_with_no_episodes(memory):
    out = memory.reflect(user_id="nobody")
    assert out == {"insights": [], "examined_episodes": 0}


# -------------------------------------------------------------------- privacy

def test_never_remember_is_honored(memory):
    res = memory.add("user: off the record, I'm interviewing at another company",
                     user_id="u1")
    assert res.get("skipped_private") is True
    assert memory.stats(user_id="u1")["total"] == 0
    assert any(e["action"] == "SKIPPED_PRIVATE"
               for e in memory.audit_log(user_id="u1"))


def test_pii_redacted_before_storage(memory):
    memory.config.redact_pii = True
    res = memory.add(
        "user: reach me at sachin@example.com, card 4111 1111 1111 1111",
        user_id="u1", infer=False,
    )
    rec = memory.get(res["episodic"])
    assert "sachin@example.com" not in rec.content
    assert "[REDACTED_EMAIL]" in rec.content
    assert "[REDACTED_CREDIT_CARD]" in rec.content
    assert res["redacted"] == {"email": 1, "credit_card": 1}
    assert any(e["action"] == "REDACT" for e in memory.audit_log(user_id="u1"))


def test_redaction_spares_dates_and_versions():
    text, counts = redact_pii(
        "meeting on 2026-06-09 at 10.30, using python 3.13.13"
    )
    assert counts == {}
    assert "2026-06-09" in text and "3.13.13" in text


def test_redaction_catches_phone_and_ssn():
    text, counts = redact_pii("call +91 98765 43210, ssn 123-45-6789")
    assert counts == {"phone": 1, "ssn": 1}
    assert "[REDACTED_PHONE]" in text and "[REDACTED_SSN]" in text
