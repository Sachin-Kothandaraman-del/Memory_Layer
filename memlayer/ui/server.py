"""Local web dashboard for memlayer (stdlib http.server — no extra deps).

Serves a single-page app plus a small JSON API:

    GET  /                      the dashboard
    GET  /api/health            version, db path, key status, stats, users
    GET  /api/memories          list or search (?q=&user=&type=&limit=)
    GET  /api/context           the prompt block for a query (?q=&user=)
    POST /api/memories          {content, user, infer, session}
    POST /api/chat              {message, history, user}
    POST /api/clear             {user}
    DELETE /api/memories/<id>

Binds to 127.0.0.1 only — this is a local tool, not a deployment server.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..core import MemoryLayer, MissingAPIKeyError
from ..middleware import MemoryMiddleware
from ..models import MemoryType

logger = logging.getLogger("memlayer")

CHAT_SYSTEM = "You are a helpful, concise assistant with long-term memory."


def _load_index() -> str:
    return (
        resources.files("memlayer.ui")
        .joinpath("index.html")
        .read_text(encoding="utf-8")
    )


class UIServer(ThreadingHTTPServer):
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
        self._middlewares: dict[str, MemoryMiddleware] = {}
        self._state_lock = threading.Lock()

    def middleware_for(self, user: str) -> MemoryMiddleware:
        with self._state_lock:
            if user not in self._middlewares:
                self._middlewares[user] = MemoryMiddleware(self.mem, user_id=user)
            return self._middlewares[user]

    def chat_fn(self) -> Callable[[list[dict]], str]:
        with self._state_lock:
            if self._chat_fn is None:
                self._chat_fn = self._build_gemini_chat()
            return self._chat_fn

    def _build_gemini_chat(self) -> Callable[[list[dict]], str]:
        key = self.mem._require_key()  # raises a friendly MissingAPIKeyError
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
                config={"system_instruction": system or CHAT_SYSTEM},
            )
            return resp.text or ""

        return chat


class Handler(BaseHTTPRequestHandler):
    server: UIServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass

    # -- routing ---------------------------------------------------------------

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._html(self.server.index_html)
            elif url.path == "/api/health":
                self._json(self._health())
            elif url.path == "/api/memories":
                self._json(self._memories(query))
            elif url.path == "/api/context":
                self._json(self._context(query))
            else:
                self._json({"error": "not found"}, 404)
        except MissingAPIKeyError as exc:
            self._json({"error": str(exc), "missing_key": True}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ui request failed")
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/memories":
                self._json(self._add(body))
            elif url.path == "/api/chat":
                self._json(self._chat(body))
            elif url.path == "/api/clear":
                user = body.get("user") or "default"
                self._json({"deleted": self.server.mem.clear(user_id=user)})
            else:
                self._json({"error": "not found"}, 404)
        except MissingAPIKeyError as exc:
            self._json({"error": str(exc), "missing_key": True}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ui request failed")
            self._json({"error": str(exc)}, 500)

    def do_DELETE(self) -> None:
        url = urlparse(self.path)
        try:
            if url.path.startswith("/api/memories/"):
                memory_id = url.path.rsplit("/", 1)[1]
                self._json({"deleted": self.server.mem.forget(memory_id)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)

    # -- endpoints ---------------------------------------------------------------

    def _health(self) -> dict:
        mem = self.server.mem
        return {
            "version": __version__,
            "db_path": mem.config.db_path,
            "has_key": bool(mem.config.resolve_api_key()),
            "llm_model": mem.config.llm_model,
            "users": mem.store.users(),
            "stats": mem.stats(),
        }

    def _memories(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["default"])[0]
        limit = int((query.get("limit") or ["50"])[0])
        type_raw = (query.get("type") or [""])[0]
        memory_type = MemoryType(type_raw) if type_raw else None
        q = (query.get("q") or [""])[0].strip()

        if q:
            try:
                hits = mem.search(
                    q, limit=limit, user_id=user, memory_type=memory_type,
                    reinforce=False,
                )
                items = []
                for hit in hits:
                    d = hit.record.to_dict()
                    d["score"] = round(hit.score, 3)
                    d["similarity"] = round(hit.similarity, 3)
                    items.append(d)
                return {"mode": "hybrid", "items": items,
                        "stats": mem.stats(user_id=user)}
            except MissingAPIKeyError:
                records = mem.store.keyword_search(
                    q, limit=limit, user_id=user, memory_type=memory_type
                )
                return {"mode": "keyword", "items": [r.to_dict() for r in records],
                        "stats": mem.stats(user_id=user)}

        records = mem.store.list(user_id=user, memory_type=memory_type, limit=limit)
        return {"mode": "list", "items": [r.to_dict() for r in records],
                "stats": mem.stats(user_id=user)}

    def _context(self, query: dict) -> dict:
        mem = self.server.mem
        user = (query.get("user") or ["default"])[0]
        q = (query.get("q") or [""])[0]
        budget = query.get("budget")
        return {
            "context": mem.get_context(
                q, user_id=user,
                token_budget=int(budget[0]) if budget else None,
            )
        }

    def _add(self, body: dict) -> dict:
        mem = self.server.mem
        return mem.add(
            body.get("content", ""),
            user_id=body.get("user") or "default",
            session_id=body.get("session"),
            infer=body.get("infer", True),
        )

    def _chat(self, body: dict) -> dict:
        mem = self.server.mem
        user = body.get("user") or "default"
        message = (body.get("message") or "").strip()
        history = body.get("history") or []
        if not message:
            return {"reply": "", "context": ""}

        chat_fn = self.server.chat_fn()
        middleware = self.server.middleware_for(user)

        context = mem.get_context(message, user_id=user)
        messages = [
            m for m in history if m.get("role") in ("user", "assistant")
        ] + [{"role": "user", "content": message}]
        augmented = (
            [{"role": "system", "content": context}] + messages
            if context else messages
        )
        reply = chat_fn(augmented)
        middleware.after(messages, reply)  # records exchange in the background
        return {"reply": reply, "context": context}

    # -- plumbing ---------------------------------------------------------------

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
    port: int = 8765,
    chat_fn: Callable[[list[dict]], str] | None = None,
) -> UIServer:
    return UIServer((host, port), mem, chat_fn=chat_fn)


def serve(
    db_path: str = "memlayer.db",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    mem = MemoryLayer.from_env(db_path=db_path)
    server = create_server(mem, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"memlayer UI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        mem.close()
