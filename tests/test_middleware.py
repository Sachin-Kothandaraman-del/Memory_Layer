from __future__ import annotations

from memlayer import MemoryMiddleware, with_memory


def _seed_fact(memory, fake_llm, fact: str, trigger: str):
    fake_llm.json_queue.append(
        {"facts": [{"content": fact, "category": "preference",
                    "importance": 0.8}]}
    )
    memory.add(trigger, user_id="u1")


def test_before_injects_system_message(memory, fake_llm):
    _seed_fact(memory, fake_llm, "User prefers concise bullet-point answers",
               "user: keep your answers concise, bullet points please")
    mw = MemoryMiddleware(memory, user_id="u1", background_writes=False)

    messages = [{"role": "user", "content": "answers about bullet points concise"}]
    augmented = mw.before(messages)

    assert augmented[0]["role"] == "system"
    assert "concise bullet-point answers" in augmented[0]["content"]
    # original list untouched
    assert messages[0]["role"] == "user"


def test_before_appends_to_existing_system_message(memory, fake_llm):
    _seed_fact(memory, fake_llm, "User's dog is named Biscuit",
               "user: my dog biscuit is a golden retriever")
    mw = MemoryMiddleware(memory, user_id="u1", background_writes=False)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what is my dog named biscuit"},
    ]
    augmented = mw.before(messages)
    assert len(augmented) == 2
    assert augmented[0]["content"].startswith("You are a helpful assistant.")
    assert "Biscuit" in augmented[0]["content"]


def test_before_no_memories_is_passthrough(memory):
    mw = MemoryMiddleware(memory, user_id="empty-user", background_writes=False)
    messages = [{"role": "user", "content": "hello there"}]
    assert mw.before(messages) == messages


def test_after_records_exchange(memory):
    mw = MemoryMiddleware(memory, user_id="u1", background_writes=False)
    mw.after(
        [{"role": "user", "content": "remind me about the demo"}],
        "Sure — the demo is on Friday.",
    )
    episodes = memory.store.list(user_id="u1")
    assert len(episodes) == 1
    assert "remind me about the demo" in episodes[0].content
    assert "demo is on Friday" in episodes[0].content
    assert episodes[0].session_id == mw.session_id


def test_wrap_full_roundtrip(memory, fake_llm):
    _seed_fact(memory, fake_llm, "User is allergic to peanuts",
               "user: I'm allergic to peanuts")
    mw = MemoryMiddleware(memory, user_id="u1", background_writes=False)

    seen = {}

    def chat_fn(messages):
        seen["messages"] = messages
        return "noted!"

    wrapped = mw.wrap(chat_fn)
    reply = wrapped([{"role": "user", "content": "peanuts allergy snack ideas"}])

    assert reply == "noted!"
    assert seen["messages"][0]["role"] == "system"
    assert "peanuts" in seen["messages"][0]["content"]
    # exchange recorded
    episodes = [
        r for r in memory.store.list(user_id="u1")
        if "snack ideas" in r.content
    ]
    assert len(episodes) == 1


def test_background_writes_flush(memory):
    mw = MemoryMiddleware(memory, user_id="u1", background_writes=True)
    mw.after([{"role": "user", "content": "background recorded line"}], "ok")
    mw.flush()
    episodes = memory.store.list(user_id="u1")
    assert any("background recorded line" in r.content for r in episodes)


def test_with_memory_decorator(memory):
    @with_memory(memory, user_id="u1", background_writes=False)
    def chat(messages):
        return "decorated reply"

    assert chat([{"role": "user", "content": "hi"}]) == "decorated reply"
    assert memory.stats(user_id="u1")["episodic"] >= 1
