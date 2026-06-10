from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from memlayer.ui.server import create_server


@pytest.fixture
def ui(memory):
    """UI server on a random port, backed by the fake-powered MemoryLayer."""
    server = create_server(
        memory, port=0, chat_fn=lambda messages: "stub reply"
    )
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


def test_serves_dashboard_page(ui):
    with urllib.request.urlopen(ui + "/") as resp:
        html = resp.read().decode("utf-8")
        assert resp.headers["Content-Type"].startswith("text/html")
    assert "memlayer" in html
    assert "chat-log" in html


def test_health_endpoint(ui):
    h = _request(ui + "/api/health")
    assert h["version"]
    assert h["stats"]["total"] == 0
    assert isinstance(h["users"], list)


def test_add_list_search_delete_roundtrip(ui, fake_llm):
    fake_llm.json_queue.append(
        {"facts": [{"content": "User plays chess on weekends",
                    "category": "preference", "importance": 0.7}]}
    )
    added = _request(ui + "/api/memories", "POST",
                     {"content": "user: I play chess every weekend",
                      "user": "u1", "infer": True})
    assert added["episodic"]
    assert len(added["facts"]) == 1

    listed = _request(ui + "/api/memories?user=u1")
    assert listed["mode"] == "list"
    assert listed["stats"]["total"] == 2  # episodic + semantic

    found = _request(ui + "/api/memories?user=u1&q=chess+weekend")
    assert found["mode"] == "hybrid"
    assert found["items"]
    assert "score" in found["items"][0]

    semantic_only = _request(ui + "/api/memories?user=u1&type=semantic")
    assert all(i["memory_type"] == "semantic" for i in semantic_only["items"])

    target = semantic_only["items"][0]["id"]
    deleted = _request(ui + f"/api/memories/{target}", "DELETE")
    assert deleted["deleted"] is True
    assert _request(ui + "/api/memories?user=u1")["stats"]["total"] == 1


def test_context_endpoint(ui, memory):
    memory.add("user: my project deadline is in march", user_id="u1",
               infer=False)
    r = _request(ui + "/api/context?user=u1&q=project+deadline+march")
    assert "deadline" in r["context"]


def test_chat_uses_stub_and_records_exchange(ui, memory):
    memory.add("user: I am a vegan cook", user_id="u1", infer=False)
    r = _request(ui + "/api/chat", "POST",
                 {"message": "vegan cook recipe ideas", "history": [],
                  "user": "u1"})
    assert r["reply"] == "stub reply"
    assert "vegan" in r["context"]  # recalled memory surfaced to the UI

    memory.flush()  # chat records the exchange in the background
    episodes = memory.store.list(user_id="u1")
    assert any("recipe ideas" in e.content for e in episodes)


def test_clear_endpoint(ui, memory):
    memory.add("user: throwaway", user_id="u1", infer=False)
    r = _request(ui + "/api/clear", "POST", {"user": "u1"})
    assert r["deleted"] == 1


def test_unknown_route_is_404(ui):
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(ui + "/api/nope")
    assert err.value.code == 404


def test_history_and_audit_endpoints(ui, fake_llm):
    fake_llm.json_queue.append(
        {"facts": [{"content": "User likes hiking", "category": "preference",
                    "importance": 0.7}]}
    )
    added = _request(ui + "/api/memories", "POST",
                     {"content": "user: I like hiking on weekends",
                      "user": "u1", "infer": True})
    fact_id = added["facts"][0]["added"][0]

    h = _request(ui + f"/api/memories/{fact_id}/history")
    assert h["record"]["id"] == fact_id
    assert h["sources"], "provenance should cite the source episode"
    assert "retention" in h

    audit = _request(ui + "/api/audit?user=u1")
    assert any(e["action"] == "ADD" for e in audit["entries"])


def test_reflect_endpoint(ui, memory, fake_llm):
    memory.add("user: a note to reflect on", user_id="u1", infer=False)
    fake_llm.json_queue.append({"insights": []})
    r = _request(ui + "/api/reflect", "POST", {"user": "u1"})
    assert r["examined_episodes"] == 1
    assert r["insights"] == []


def test_chat_returns_recalled_breakdown(ui, memory):
    memory.add("user: I love climbing in the alps", user_id="u1", infer=False)
    r = _request(ui + "/api/chat", "POST",
                 {"message": "climbing alps tips", "history": [], "user": "u1"})
    assert r["recalled"], "glass-box recall list should be returned"
    item = r["recalled"][0]
    for key in ("similarity", "retention", "importance", "strength", "score"):
        assert key in item
