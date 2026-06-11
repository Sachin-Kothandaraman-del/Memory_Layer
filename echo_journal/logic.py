"""Echo's endpoint logic, independent of any HTTP server.

Shared by the local app (echo_journal.server, stdlib HTTP + SQLite) and the
cloud deployment (api/index.py on Vercel + Supabase) so both run identical
behavior. Every function takes the MemoryLayer (and a chat_fn where needed)
and returns a JSON-serializable dict.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

from memlayer import MemoryLayer, MemoryType
from memlayer.models import AuditEntry

RESCUE_THRESHOLD = 0.35   # below this retention a memory shows up in Rescue
RESCUE_BOOST = 3.0        # strength multiplier when the user keeps a memory

ECHO_SYSTEM = """\
You are Echo, a warm, attentive journaling companion with long-term memory.
The user is writing in their private journal; you are the quiet voice that
listens. Respond briefly (2-4 sentences): reflect back what you heard,
connect it to relevant past memories when they're provided, and occasionally
ask ONE gentle follow-up question. Never lecture, never therapize, never
diagnose. You are a companion, not a coach."""

PASTSELF_SYSTEM = """\
You are the user's memory exactly as it existed on {date}. You may ONLY use
the memories provided below — they are everything that was known and believed
on that day. Answer in second person about what they knew, felt, wanted, or
believed back then ("Back then, you were..."). Do not use any later
knowledge. If the memories don't cover the question, say so honestly and
briefly. Keep it to 2-5 sentences."""

ChatFn = Callable[[list[dict]], str]


def state(mem: MemoryLayer, user: str) -> dict:
    episodes = mem.store.list(
        user_id=user, memory_type=MemoryType.EPISODIC, limit=10_000
    )
    days = {e.session_id for e in episodes if e.session_id}
    today = dt.date.today().isoformat()
    return {
        "has_key": bool(mem.config.resolve_api_key()),
        "stats": mem.stats(user_id=user),
        "journal_days": len(days),
        "entries_today": sum(1 for e in episodes if e.session_id == today),
        "fading_count": len(_fading(mem, user)),
        "today": today,
    }


def _fading(mem: MemoryLayer, user: str) -> list[tuple[float, object]]:
    fading = []
    for rec in mem.store.list(user_id=user, limit=5000):
        retention = mem.retention(rec)
        if retention < RESCUE_THRESHOLD:
            fading.append((retention, rec))
    fading.sort(key=lambda pair: pair[0])
    return fading


def rescue(mem: MemoryLayer, user: str) -> dict:
    items = []
    for retention, rec in _fading(mem, user)[:30]:
        d = rec.to_dict()
        d["retention"] = round(retention, 3)
        items.append(d)
    return {"items": items}


def keep(mem: MemoryLayer, user: str, memory_id: str) -> dict:
    rec = mem.store.get(memory_id)
    if rec is None or rec.user_id != user:
        return {"error": "not found"}
    mem.store.touch(
        [memory_id],
        strength_factor=RESCUE_BOOST,
        strength_max=mem.config.strength_max,
    )
    mem.store.log_audit(AuditEntry(
        action="RESCUE", user_id=rec.user_id, memory_id=memory_id,
        reasoning="user chose to keep this fading memory",
    ))
    rec = mem.store.get(memory_id)
    return {
        "kept": True,
        "strength": round(rec.strength, 2),
        "retention": round(mem.retention(rec), 3),
    }


def letgo(mem: MemoryLayer, user: str, memory_id: str) -> dict:
    rec = mem.store.get(memory_id)
    if rec is None or rec.user_id != user:
        return {"deleted": False}
    return {"deleted": mem.forget(memory_id)}


def entry(mem: MemoryLayer, chat_fn: ChatFn, user: str,
          text: str, history: list[dict]) -> dict:
    text = (text or "").strip()
    if not text:
        return {"reply": "", "recalled": []}

    # recall happens BEFORE the entry is stored, so the context is
    # genuinely "what Echo already knew"
    context, recalled = mem.build_context(text, user_id=user)
    messages: list[dict] = [{"role": "system", "content": ECHO_SYSTEM}]
    if context:
        messages.append({"role": "system", "content": context})
    messages += [
        m for m in (history or []) if m.get("role") in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": text})
    reply = chat_fn(messages)

    result = mem.add(
        text,
        user_id=user,
        session_id=dt.date.today().isoformat(),
    )
    return {
        "reply": reply,
        "recalled": _recalled(recalled),
        "entry_id": result.get("episodic"),
        "facts": result.get("facts", []),
        "skipped_private": result.get("skipped_private", False),
        "redacted": result.get("redacted"),
    }


def pastself(mem: MemoryLayer, chat_fn: ChatFn, user: str,
             date_str: str, question: str) -> dict:
    question = (question or "").strip()
    if not date_str or not question:
        return {"answer": "", "recalled": [], "date": date_str}
    day = dt.datetime.strptime(date_str, "%Y-%m-%d")
    as_of = (day + dt.timedelta(days=1)).timestamp()  # end of that day

    context, recalled = mem.build_context(question, user_id=user, as_of=as_of)
    if not context:
        return {
            "answer": "I don't have any memories from back then that "
                      "speak to that.",
            "recalled": [], "date": date_str,
        }
    messages = [
        {"role": "system", "content": PASTSELF_SYSTEM.format(date=date_str)},
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]
    answer = chat_fn(messages)
    return {
        "answer": answer,
        "recalled": _recalled(recalled),
        "date": date_str,
    }


def onthisday(mem: MemoryLayer, user: str) -> dict:
    today = dt.date.today()
    items = []
    for rec in mem.store.list(
        user_id=user, memory_type=MemoryType.EPISODIC, limit=10_000
    ):
        then = dt.date.fromtimestamp(rec.created_at)
        if then >= today or then.day != today.day:
            continue
        months = (today.year - then.year) * 12 + (today.month - then.month)
        if months < 1:
            continue
        d = rec.to_dict()
        d["ago"] = (f"{months // 12} year{'s' if months >= 24 else ''} ago"
                    if months >= 12
                    else f"{months} month{'s' if months > 1 else ''} ago")
        items.append(d)
    items.sort(key=lambda d: d["created_at"], reverse=True)
    return {"items": items[:10]}


def insights(mem: MemoryLayer, user: str) -> dict:
    records = mem.store.list(
        user_id=user, memory_type=MemoryType.SEMANTIC, limit=1000
    )
    items = []
    for rec in records:
        if rec.category != "insight":
            continue
        d = rec.to_dict()
        d["evidence_count"] = len(rec.source_ids)
        items.append(d)
    return {"items": items}


def story(mem: MemoryLayer, user: str, memory_id: str) -> dict | None:
    rec = mem.store.get(memory_id)
    if rec is None or rec.user_id != user:
        return None
    return mem.history(memory_id)


def _recalled(scored) -> list[dict]:
    return [
        {
            "id": s.record.id,
            "content": s.record.content,
            "memory_type": s.record.memory_type.value,
            "score": round(s.score, 3),
            "retention": round(s.recency, 3),
        }
        for s in scored
    ]
