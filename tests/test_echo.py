"""Tests for the Echo journal app's endpoints (fake-backed, no API key)."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from echo_journal.server import RESCUE_THRESHOLD, create_server
from memlayer.models import MemoryType

DAY = 86400.0


@pytest.fixture
def echo(memory):
    server = create_server(memory, port=0,
                           chat_fn=lambda messages: "echo stub reply")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base
    server.shutdown()


def _request(url: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _backdate(memory, memory_id: str, days: float) -> None:
    rec = memory.get(memory_id)
    rec.created_at = rec.updated_at = time.time() - days * DAY
    rec.last_accessed_at = rec.updated_at
    memory.store.add(rec)


def test_serves_journal_page(echo):
    with urllib.request.urlopen(echo + "/") as resp:
        html = resp.read().decode("utf-8")
    assert "echo" in html and "Rescue" in html


def test_entry_stores_and_replies(echo, memory):
    r = _request(echo + "/api/entry", "POST",
                 {"text": "Long day, but the demo went well.", "history": [],
                  "user": "me"})
    assert r["reply"] == "echo stub reply"
    assert r["entry_id"]
    rec = memory.get(r["entry_id"])
    assert rec.memory_type == MemoryType.EPISODIC
    assert rec.session_id == dt.date.today().isoformat()

    state = _request(echo + "/api/state?user=me")
    assert state["entries_today"] == 1
    assert state["journal_days"] == 1


def test_entry_recalls_before_storing(echo, memory):
    memory.add("the demo presentation is on friday", user_id="me", infer=False)
    r = _request(echo + "/api/entry", "POST",
                 {"text": "nervous about the demo presentation", "history": [],
                  "user": "me"})
    assert any("friday" in m["content"] for m in r["recalled"])


def test_rescue_keep_reinforces(echo, memory):
    res = memory.add("that perfect evening walk by the river", user_id="me",
                     infer=False)
    _backdate(memory, res["episodic"], days=60)

    rescue = _request(echo + "/api/rescue?user=me")
    assert len(rescue["items"]) == 1
    item = rescue["items"][0]
    assert item["retention"] < RESCUE_THRESHOLD

    kept = _request(echo + "/api/keep", "POST", {"id": item["id"]})
    assert kept["kept"] is True
    assert kept["strength"] > 1.0
    assert kept["retention"] > item["retention"]  # rescued: vivid again

    # rescue is audited (glass box all the way down)
    assert any(e["action"] == "RESCUE" for e in memory.audit_log(user_id="me"))
    # no longer fading
    assert _request(echo + "/api/rescue?user=me")["items"] == []


def test_letgo_deletes(echo, memory):
    res = memory.add("a memory to release", user_id="me", infer=False)
    r = _request(echo + "/api/letgo", "POST", {"id": res["episodic"]})
    assert r["deleted"] is True
    assert memory.stats(user_id="me")["total"] == 0


def test_pastself_answers_from_then(echo, memory, fake_llm):
    # belief on day -30: Berlin; superseded today by Munich
    memory.config.consolidation_sim_threshold = 0.5
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Berlin", "category": "identity",
                    "importance": 0.9}]}
    )
    memory.add("user: I live in Berlin", user_id="me")
    for rec in memory.store.list(user_id="me", current_only=False):
        _backdate(memory, rec.id, days=30)
        rec = memory.get(rec.id)
        rec.valid_from = rec.created_at
        memory.store.add(rec)
    old = memory.store.list(user_id="me", memory_type=MemoryType.SEMANTIC)[0]
    fake_llm.json_queue.append(
        {"facts": [{"content": "User lives in Munich", "category": "identity",
                    "importance": 0.9}]}
    )
    fake_llm.json_queue.append(
        {"operations": [{"op": "UPDATE", "id": old.id,
                         "content": "User lives in Munich",
                         "reasoning": "moved"}]}
    )
    memory.add("user: I moved to Munich", user_id="me")

    past_date = dt.date.fromtimestamp(time.time() - 15 * DAY).isoformat()
    r = _request(echo + f"/api/pastself?user=me&date={past_date}"
                        "&q=where+do+I+live+berlin+munich")
    assert r["answer"] == "echo stub reply"
    contents = " ".join(m["content"] for m in r["recalled"])
    assert "Berlin" in contents
    assert "Munich" not in contents  # the future must not leak into the past


def test_pastself_with_no_memories(echo):
    r = _request(echo + "/api/pastself?user=nobody&date=2020-01-01&q=anything")
    assert "don't have any memories" in r["answer"]


def test_onthisday_finds_anniversaries(echo, memory):
    res = memory.add("user: started learning the piano today", user_id="me",
                     infer=False)
    # exactly this calendar day, one month-ish back: subtract until day matches
    rec = memory.get(res["episodic"])
    today = dt.date.today()
    then = (today - dt.timedelta(days=27))
    while then.day != today.day:
        then -= dt.timedelta(days=1)
    ts = time.mktime(then.timetuple()) + 3600
    rec.created_at = rec.updated_at = rec.last_accessed_at = ts
    memory.store.add(rec)

    r = _request(echo + "/api/onthisday?user=me")
    assert len(r["items"]) == 1
    assert "piano" in r["items"][0]["content"]
    assert "ago" in r["items"][0]


def test_insights_and_reflect(echo, memory, fake_llm):
    e1 = memory.add("user: skipped the gym, too tired", user_id="me",
                    infer=False)
    e2 = memory.add("user: tired again, skipped the run", user_id="me",
                    infer=False)
    fake_llm.json_queue.append(
        {"insights": [{
            "insight": "User's energy dips are crowding out exercise",
            "evidence": [e1["episodic"], e2["episodic"]],
            "confidence": 0.7,
            "reasoning": "two entries pair tiredness with skipped workouts",
        }]}
    )
    r = _request(echo + "/api/reflect", "POST", {"user": "me"})
    assert len(r["insights"]) == 1

    insights = _request(echo + "/api/insights?user=me")
    assert len(insights["items"]) == 1
    item = insights["items"][0]
    assert item["evidence_count"] == 2

    story = _request(echo + f"/api/story/{item['id']}")
    assert len(story["sources"]) == 2
    assert any("gym" in s["content"] for s in story["sources"])


def test_off_the_record_entry(echo, memory):
    r = _request(echo + "/api/entry", "POST",
                 {"text": "off the record, don't keep this one", "history": [],
                  "user": "me"})
    assert r["skipped_private"] is True
    assert memory.stats(user_id="me")["total"] == 0
