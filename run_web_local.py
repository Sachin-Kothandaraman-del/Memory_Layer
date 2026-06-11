"""Run the Echo WEBSITE (the version with account creation) on your machine.

This serves public/index.html plus the exact same /api code that runs on
Vercel — so you can see and test the sign-in / create-account flow locally
before deploying.

Usage:
    python run_web_local.py            # http://127.0.0.1:8790

Requirements (same as the deployed site — see .env.example / DEPLOY.md):
    .env containing SUPABASE_URL, SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY
    and the schema from supabase/schema.sql applied to your Supabase project.

Without those the page still loads (you can see the account screen), but
signing in and journaling will not work yet.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memlayer.config import load_dotenv_file  # noqa: E402

load_dotenv_file()

# load the Vercel function as a module (api/ is not a package by design)
_spec = importlib.util.spec_from_file_location(
    "vercel_api", ROOT / "api" / "index.py"
)
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)

INDEX = (ROOT / "public" / "index.html").read_bytes()


class LocalWebHandler(api.handler):
    """Same /api behavior as Vercel; everything else serves the site."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            super().do_GET()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(INDEX)))
        self.end_headers()
        self.wfile.write(INDEX)


def main() -> int:
    required = ("SUPABASE_URL", "SUPABASE_ANON_KEY",
                "SUPABASE_SERVICE_ROLE_KEY", "GEMINI_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("[!] Missing in .env:", ", ".join(missing))
        print("    The page will load so you can see the account screen,")
        print("    but sign-in/journaling needs these set - see DEPLOY.md.\n")

    port = int(os.environ.get("PORT", "8790"))
    server = ThreadingHTTPServer(("127.0.0.1", port), LocalWebHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Echo website (with accounts) running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
