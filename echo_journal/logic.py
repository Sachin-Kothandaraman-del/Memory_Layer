"""Echo's endpoint logic, independent of any HTTP server.

Shared by the local app (echo_journal.server, stdlib HTTP + SQLite) and the
cloud deployment (api/index.py on Vercel + Supabase) so both run identical
behavior. Every function takes the MemoryLayer (and a chat_fn where needed)
and returns a JSON-serializable dict.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Callable

from memlayer import MemoryLayer, MemoryType
from memlayer.models import AuditEntry

logger = logging.getLogger("echo")

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


def today(mem: MemoryLayer, user: str) -> dict:
    """Today's journal entries, oldest first — so a reload shows the day so far."""
    day = dt.date.today().isoformat()
    episodes = mem.store.list(
        user_id=user, session_id=day,
        memory_type=MemoryType.EPISODIC, limit=500,
    )
    episodes.sort(key=lambda r: r.created_at)
    return {
        "items": [
            {"id": e.id, "content": e.content, "created_at": e.created_at}
            for e in episodes
        ]
    }


def entries_in_last_hour(mem: MemoryLayer, user: str) -> int:
    """How many entries this user wrote in the past hour (for rate limiting)."""
    cutoff = time.time() - 3600.0
    recent = mem.store.list(
        user_id=user, memory_type=MemoryType.EPISODIC,
        limit=300, current_only=False,
    )
    return sum(1 for r in recent if r.created_at >= cutoff)


def weekly_reflection(
    mem: MemoryLayer,
    max_users: int = 3,
    active_days: float = 7.0,
    cooldown_days: float = 6.0,
) -> dict:
    """Run a reflection pass for users active this week (for the cron job).

    Caps work per invocation (serverless time budget) and records a
    REFLECT_RUN audit entry per user so retries within ``cooldown_days``
    don't re-reflect the same people.
    """
    now = time.time()
    reflected = 0
    insights_created = 0
    for user in mem.store.users():
        if reflected >= max_users:
            break
        latest = mem.store.list(
            user_id=user, memory_type=MemoryType.EPISODIC, limit=1
        )
        if not latest or latest[0].updated_at < now - active_days * 86400.0:
            continue  # not active this week
        ran_recently = any(
            e.action == "REFLECT_RUN" and e.ts >= now - cooldown_days * 86400.0
            for e in mem.store.get_audit(user_id=user, limit=50)
        )
        if ran_recently:
            continue
        try:
            result = mem.reflect(user_id=user)
        except Exception as exc:  # noqa: BLE001 - one user must not kill the run
            logger.error("weekly reflection failed for %s: %s", user, exc)
            continue
        mem.store.log_audit(AuditEntry(
            action="REFLECT_RUN", user_id=user,
            reasoning="scheduled weekly reflection",
            detail={"insights": len(result["insights"]),
                    "examined": result["examined_episodes"]},
        ))
        reflected += 1
        insights_created += len(result["insights"])
    return {"reflected_users": reflected, "insights_created": insights_created}


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
