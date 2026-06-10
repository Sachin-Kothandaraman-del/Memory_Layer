"""Echo's local server: journal endpoints layered on a memlayer MemoryLayer.

    GET  /                  the app
    GET  /api/state         streak, stats, fading count, key status
    POST /api/entry         {text, history, user} -> companion reply + memory ops
    GET  /api/rescue        memories about to fade (retention ascending)
    POST /api/keep          {id} -> reinforce a fading memory (rescue it)
    POST /api/letgo         {id} -> hard-delete a memory (audited)
    GET  /api/pastself      ?date=YYYY-MM-DD&q=... -> answer from that day's beliefs
    GET  /api/onthisday     entries from this day in earlier months/years
    GET  /api/insights      reflection insights (evidence-cited)
    POST /api/reflect       {user} -> run a reflection pass now
    GET  /api/story/<id>    one memory's full story (versions/sources/audit)

Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable
from urllib.parse import parse_qs, urlparse

from memlayer import MemoryLayer, MemoryType
from memlayer.config import MemoryConfig, load_dotenv_file
from memlayer.core import MISSING_KEY_MESSAGE, MissingAPIKeyError
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


def _load_index() -> str:
    return (
        resources.files("echo_journal")
        .joinpath("index.html")
        .read_text(encoding="utf-8")
    )


class EchoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        mem: MemoryLayer,
        chat_fn: Callable[[list[dict]], str] | None = None,
    ):
        super().__init__(address, Handler)
        self.mem = mem
        self.index_html = _load_index()
        self._chat_fn = chat_fn
        self._state_lock = threading.Lock()

    def chat_fn(self) -> Callable[[list[dict]], str]:
        with self._state_lock:
            if self._chat_fn is None:
                self._chat_fn = self._build_gemini_chat()
            return self._chat_fn

    def _build_gemini_chat(self) -> Callable[[list[dict]], str]:
        key = self.mem._require_key()
        from google import genai

        client = genai.Client(api_key=key) if key else genai.Client()
        model = self.mem.config.llm_model

        def chat(messages: list[dict]) -> str:
            system = "\n\n".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            contents = [
                {"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]}
                for m in messages
                if m["role"] in ("user", "assistant")
            ]
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config={"system_instruction": system or ECHO_SYSTEM},
            )
            return resp.text or ""

        return chat


class Handler(BaseHTTPRequestHandler):
    server: EchoServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._html(self.server.index_html)
            elif url.path == "/api/state":
                self._json(self._state(query))
            elif url.path == "/api/rescue":
                self._json(self._rescue(query))
            elif url.path == "/api/pastself":
                self._json(self._pastself(query))
            elif url.path == "/api/onthisday":
                self._json(self._onthisday(query))
            elif url.path == "/api/insights":
                self._json(self._insights(query))
            elif url.path.startswith("/api/story/"):
                memory_id = url.path.rsplit("/", 1)[1]
                h = self.server.mem.history(memory_id)
                self._json(h if h else {"error": "not found"},
                           200 if h else 404)
            else:
                self._json({"error": "not found"}, 404)
        except MissingAPIKeyError as exc:
            self._json({"error": str(exc), "missing_key": True}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("echo request failed")
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/entry":
                self._json(self._entry(body))
            elif url.path == "/api/keep":
                self._json(self._keep(body))
            elif url.path == "/api/letgo":
                memory_id = body.get("id", "")
                self._json({"deleted": self.server.mem.forget(memory_id)})
            elif url.path == "/api/reflect":
                user = body.get("user") or "me"
                self._json(self.server.mem.reflect(user_id=user))
            else:
                self._json({"error": "not found"}, 404)
        except MissingAPIKeyError as exc:
            self._json({"error": str(exc), "missing_key": True}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("echo request failed")
            self._json({"error": str(exc)}, 500)

    # -- endpoints ----------------------------------------------------------------

    def _state(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
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
            "fading_count": len(self._fading(user)),
            "today": today,
        }

    def _fading(self, user: str) -> list[tuple[float, object]]:
        mem = self.server.mem
        fading = []
        for rec in mem.store.list(user_id=user, limit=5000):
            retention = mem.retention(rec)
            if retention < RESCUE_THRESHOLD:
                fading.append((retention, rec))
        fading.sort(key=lambda pair: pair[0])
        return fading

    def _rescue(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
        items = []
        for retention, rec in self._fading(user)[:30]:
            d = rec.to_dict()
            d["retention"] = round(retention, 3)
            items.append(d)
        return {"items": items}

    def _keep(self, body: dict) -> dict:
        mem = self.server.mem
        memory_id = body.get("id", "")
        rec = mem.store.get(memory_id)
        if rec is None:
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

    def _entry(self, body: dict) -> dict:
        mem = self.server.mem
        user = body.get("user") or "me"
        text = (body.get("text") or "").strip()
        history = body.get("history") or []
        if not text:
            return {"reply": "", "recalled": []}

        # recall happens BEFORE the entry is stored, so the context is
        # genuinely "what Echo already knew"
        context, recalled = mem.build_context(text, user_id=user)
        messages: list[dict] = [{"role": "system", "content": ECHO_SYSTEM}]
        if context:
            messages.append({"role": "system", "content": context})
        messages += [
            m for m in history if m.get("role") in ("user", "assistant")
        ]
        messages.append({"role": "user", "content": text})
        reply = self.server.chat_fn()(messages)

        result = mem.add(
            text,
            user_id=user,
            session_id=dt.date.today().isoformat(),
        )
        return {
            "reply": reply,
            "recalled": self._recalled(recalled),
            "entry_id": result.get("episodic"),
            "facts": result.get("facts", []),
            "skipped_private": result.get("skipped_private", False),
            "redacted": result.get("redacted"),
        }

    def _pastself(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
        date_str = (query.get("date") or [""])[0]
        question = (query.get("q") or [""])[0].strip()
        if not date_str or not question:
            return {"answer": "", "recalled": [], "date": date_str}
        day = dt.datetime.strptime(date_str, "%Y-%m-%d")
        as_of = (day + dt.timedelta(days=1)).timestamp()  # end of that day

        context, recalled = mem.build_context(
            question, user_id=user, as_of=as_of
        )
        if not context:
            return {
                "answer": "I don't have any memories from back then that "
                          "speak to that.",
                "recalled": [], "date": date_str,
            }
        messages = [
            {"role": "system",
             "content": PASTSELF_SYSTEM.format(date=date_str)},
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        answer = self.server.chat_fn()(messages)
        return {
            "answer": answer,
            "recalled": self._recalled(recalled),
            "date": date_str,
        }

    def _onthisday(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
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

    def _insights(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
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

    @staticmethod
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

    # -- plumbing ----------------------------------------------------------------

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def create_server(
    mem: MemoryLayer,
    host: str = "127.0.0.1",
    port: int = 8780,
    chat_fn: Callable[[list[dict]], str] | None = None,
) -> EchoServer:
    return EchoServer((host, port), mem, chat_fn=chat_fn)


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    parser = argparse.ArgumentParser(
        prog="echo-journal",
        description="Echo - a journal you talk to, that lets you talk to "
                    "your past self.",
    )
    parser.add_argument("--db", default=os.environ.get(
        "ECHO_DB_PATH", "echo.db"))
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    config = MemoryConfig.from_env(db_path=args.db)
    config.redact_pii = True  # a journal should be private by default
    mem = MemoryLayer(config=config)
    server = create_server(mem, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Echo is listening at {url}  (Ctrl+C to close the journal)")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nClosing the journal...")
    finally:
        server.shutdown()
        mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
