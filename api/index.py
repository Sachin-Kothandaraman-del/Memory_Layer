"""Echo cloud API — a single Vercel Python serverless function.

Routes (all under /api/, same contract as the local echo_journal server):
    GET  /api/config       public Supabase config for the frontend (no auth)
    GET  /api/state        POST /api/entry      GET  /api/rescue
    POST /api/keep         POST /api/letgo      GET  /api/pastself
    GET  /api/onthisday    GET  /api/insights   POST /api/reflect
    GET  /api/story/<id>

Auth: every endpoint except /api/config requires a Supabase access token
(``Authorization: Bearer <jwt>``). The authenticated Supabase user id IS the
memlayer user_id — client-supplied user values are ignored, so accounts are
hard-isolated.

Required environment variables (set in Vercel project settings):
    SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
# Append project root so local app modules are importable, but do not shadow
# installed third-party packages (e.g., `supabase`) with same-named folders.
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from echo_journal import logic  # noqa: E402
from echo_journal.server import build_gemini_chat  # noqa: E402
from memlayer import MemoryConfig, MemoryLayer  # noqa: E402
from memlayer.core import MissingAPIKeyError  # noqa: E402
from memlayer.storage.supabase_store import SupabaseMemoryStore  # noqa: E402

logger = logging.getLogger("echo-cloud")

# Module-level singletons survive across requests on a warm function instance
_mem: MemoryLayer | None = None
_chat_fn = None


def get_mem() -> MemoryLayer:
    global _mem
    if _mem is None:
        config = MemoryConfig.from_env()
        config.redact_pii = True  # a journal is private by default
        _mem = MemoryLayer(config=config, store=SupabaseMemoryStore())
    return _mem


def get_chat_fn():
    global _chat_fn
    if _chat_fn is None:
        _chat_fn = build_gemini_chat(get_mem())
    return _chat_fn


def authenticate(headers) -> str | None:
    """Resolve the Supabase access token to a user id (None = unauthorized)."""
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        return None
    token = auth[len("Bearer "):].strip()
    if not token:
        return None

    base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    api_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    )
    if not base_url or not api_key:
        return None

    req = urlrequest.Request(
        f"{base_url}/auth/v1/user",
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("id")
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError):
        return None


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel naming convention
    def log_message(self, fmt: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/config":
                self._json({
                    "supabase_url": os.environ.get("SUPABASE_URL", ""),
                    "supabase_anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
                })
                return
            user = authenticate(self.headers)
            if user is None:
                self._json({"error": "not signed in"}, 401)
                return
            mem = get_mem()
            if url.path == "/api/state":
                self._json(logic.state(mem, user))
            elif url.path == "/api/rescue":
                self._json(logic.rescue(mem, user))
            elif url.path == "/api/pastself":
                self._json(logic.pastself(
                    mem, get_chat_fn(), user,
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
            logger.exception("echo cloud request failed")
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            user = authenticate(self.headers)
            if user is None:
                self._json({"error": "not signed in"}, 401)
                return
            body = self._body()
            mem = get_mem()
            if url.path == "/api/entry":
                self._json(logic.entry(
                    mem, get_chat_fn(), user,
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
            logger.exception("echo cloud request failed")
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
