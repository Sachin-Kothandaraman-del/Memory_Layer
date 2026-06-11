"""Echo's local server: journal endpoints layered on a memlayer MemoryLayer.

The endpoint logic itself lives in :mod:`echo_journal.logic` and is shared
with the cloud deployment (api/index.py on Vercel + Supabase).

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
import json
import logging
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Callable
from urllib.parse import parse_qs, urlparse

from memlayer import MemoryLayer
from memlayer.config import MemoryConfig, load_dotenv_file
from memlayer.core import MissingAPIKeyError

from . import logic
from .logic import ECHO_SYSTEM, PASTSELF_SYSTEM, RESCUE_BOOST, RESCUE_THRESHOLD

__all__ = [
    "EchoServer", "create_server", "main",
    "ECHO_SYSTEM", "PASTSELF_SYSTEM", "RESCUE_BOOST", "RESCUE_THRESHOLD",
]

logger = logging.getLogger("echo")


def _load_index() -> str:
    return (
        resources.files("echo_journal")
        .joinpath("index.html")
        .read_text(encoding="utf-8")
    )


def build_gemini_chat(mem: MemoryLayer) -> Callable[[list[dict]], str]:
    """Plain Gemini chat function used by both local and cloud deployments."""
    key = mem._require_key()
    from google import genai

    client = genai.Client(api_key=key) if key else genai.Client()
    model = mem.config.llm_model

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
                self._chat_fn = build_gemini_chat(self.mem)
            return self._chat_fn


class Handler(BaseHTTPRequestHandler):
    server: EchoServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        mem = self.server.mem
        user = (query.get("user") or ["me"])[0]
        try:
            if url.path in ("/", "/index.html"):
                self._html(self.server.index_html)
            elif url.path == "/api/state":
                self._json(logic.state(mem, user))
            elif url.path == "/api/today":
                self._json(logic.today(mem, user))
            elif url.path == "/api/rescue":
                self._json(logic.rescue(mem, user))
            elif url.path == "/api/pastself":
                self._json(logic.pastself(
                    mem, self.server.chat_fn(), user,
                    (query.get("date") or [""])[0],
                    (query.get("q") or [""])[0],
                ))
            elif url.path == "/api/onthisday":
                self._json(logic.onthisday(mem, user))
            elif url.path == "/api/insights":
                self._json(logic.insights(mem, user))
            elif url.path.startswith("/api/story/"):
                memory_id = url.path.rsplit("/", 1)[1]
                h = logic.story(mem, user, memory_id)
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
        mem = self.server.mem
        try:
            body = self._body()
            user = body.get("user") or "me"
            if url.path == "/api/entry":
                self._json(logic.entry(
                    mem, self.server.chat_fn(), user,
                    body.get("text", ""), body.get("history") or [],
                ))
            elif url.path == "/api/keep":
                self._json(logic.keep(mem, user, body.get("id", "")))
            elif url.path == "/api/letgo":
                self._json(logic.letgo(mem, user, body.get("id", "")))
            elif url.path == "/api/reflect":
                self._json(mem.reflect(user_id=user))
            else:
                self._json({"error": "not found"}, 404)
        except MissingAPIKeyError as exc:
            self._json({"error": str(exc), "missing_key": True}, 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("echo request failed")
            self._json({"error": str(exc)}, 500)

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
