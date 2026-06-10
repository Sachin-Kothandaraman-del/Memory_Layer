"""memlayer command-line interface.

Manage memories without writing code, run health checks, and chat with a
memory-augmented Gemini agent straight from the terminal.

Store-only commands (stats, list, forget, clear, export, prune) work without
an API key; commands that embed or extract (add, search, context, chat,
import) need GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

from . import __version__
from .config import MemoryConfig, load_dotenv_file
from .models import MemoryRecord, MemoryType

CHAT_SYSTEM = "You are a helpful, concise assistant with long-term memory."


# --------------------------------------------------------------------- helpers

def _db_path(args) -> str:
    return args.db or os.environ.get("MEMLAYER_DB_PATH", "memlayer.db")


def _open_store(args):
    from .storage.sqlite_store import SQLiteMemoryStore

    return SQLiteMemoryStore(_db_path(args))


def _open_memory(args):
    from .core import MemoryLayer

    return MemoryLayer.from_env(db_path=_db_path(args))


def _mtype(value: str | None) -> MemoryType | None:
    return MemoryType(value) if value else None


def _fmt_date(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _print_record(rec: MemoryRecord, score: float | None = None) -> None:
    prefix = f"{score:.3f}  " if score is not None else ""
    print(
        f"{prefix}{rec.id[:8]}  [{rec.memory_type.value:8}]  "
        f"{_fmt_date(rec.updated_at)}  {rec.content}"
    )


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# -------------------------------------------------------------------- commands

def cmd_init(args) -> int:
    """One-time setup: save the API key to .env, then run the health check."""
    existing = MemoryConfig.from_env().resolve_api_key()
    if existing:
        print("An API key is already configured.")
    else:
        print("Get a free Gemini API key at: https://aistudio.google.com/apikey")
        try:
            key = input("Paste your API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if not key:
            print("No key entered - nothing changed.")
            return 1
        _write_env_key(key)
        os.environ["GEMINI_API_KEY"] = key
        print("Saved to .env\n")
    return cmd_doctor(args)


def _write_env_key(key: str, path: str = ".env") -> None:
    lines: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:
            lines = [
                line.rstrip("\n")
                for line in fh
                if not line.strip().startswith("GEMINI_API_KEY=")
            ]
    lines.append(f"GEMINI_API_KEY={key}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def cmd_doctor(args) -> int:
    """Check that everything is ready to use."""
    all_ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        if not passed:
            all_ok = False
        mark = "[ok]" if passed else "[!!]"
        print(f"  {mark} {label}" + (f": {detail}" if detail else ""))

    print(f"memlayer {__version__} health check\n")

    check("python", sys.version_info >= (3, 10),
          f"{sys.version_info.major}.{sys.version_info.minor}")
    try:
        import numpy

        check("numpy", True, numpy.__version__)
    except ImportError:
        check("numpy", False, "missing - pip install numpy")
    try:
        import google.genai as _genai  # noqa: F401

        check("google-genai", True)
    except ImportError:
        check("google-genai", False, "missing - pip install google-genai")

    from .storage.sqlite_store import SQLiteMemoryStore

    probe = SQLiteMemoryStore(":memory:")
    check("sqlite FTS5 (keyword search)", probe._fts_enabled,
          "" if probe._fts_enabled else "unavailable - vector search only")
    probe.close()

    key = MemoryConfig.from_env().resolve_api_key()
    check(
        "gemini api key", bool(key),
        "configured" if key
        else "missing - run 'memlayer init' or set GEMINI_API_KEY "
             "(https://aistudio.google.com/apikey)",
    )

    db = _db_path(args)
    if os.path.exists(db):
        store = SQLiteMemoryStore(db)
        check("database", True, f"{db} ({store.count()} memories)")
        store.close()
    else:
        check("database", True, f"{db} (will be created on first write)")

    if getattr(args, "live", False):
        if not key:
            check("live api call", False, "skipped - no api key")
        else:
            import time as _time

            from .embeddings import GeminiEmbedder
            from .llm import GeminiLLM

            cfg = MemoryConfig.from_env()
            try:
                t0 = _time.time()
                GeminiEmbedder(api_key=key, model=cfg.embed_model,
                               dim=cfg.embed_dim).embed_query("ping")
                check("live embedding", True, f"{_time.time() - t0:.2f}s")
                t0 = _time.time()
                GeminiLLM(api_key=key, model=cfg.llm_model).generate(
                    "Reply with the single word: pong"
                )
                check("live generation", True, f"{_time.time() - t0:.2f}s")
            except Exception as exc:  # noqa: BLE001
                check("live api call", False, str(exc))

    print("\nAll good - try: memlayer chat" if all_ok
          else "\nFix the [!!] items above, then re-run: memlayer doctor")
    return 0 if all_ok else 1


def cmd_add(args) -> int:
    mem = _open_memory(args)
    try:
        result = mem.add(
            args.text,
            user_id=args.user,
            session_id=args.session,
            infer=not args.no_infer,
        )
        print(f"Stored episodic memory {result['episodic'][:8]}")
        for fact in result["facts"]:
            actions = []
            if fact["added"]:
                actions.append("added")
            if fact["updated"]:
                actions.append("updated existing")
            if fact["deleted"]:
                actions.append(f"deleted {len(fact['deleted'])} obsolete")
            if fact["skipped"] and not actions:
                actions.append("already known")
            print(f"  fact ({', '.join(actions)}): {fact['content']}")
        if not result["facts"] and not args.no_infer:
            print("  (no durable facts extracted)")
    finally:
        mem.close()
    return 0


def cmd_search(args) -> int:
    mem = _open_memory(args)
    try:
        hits = mem.search(
            args.query, limit=args.limit, user_id=args.user,
            memory_type=_mtype(args.type),
        )
        if not hits:
            print("No matching memories.")
            return 0
        for hit in hits:
            _print_record(hit.record, hit.score)
    finally:
        mem.close()
    return 0


def cmd_context(args) -> int:
    mem = _open_memory(args)
    try:
        block = mem.get_context(
            args.query, user_id=args.user, token_budget=args.budget
        )
        print(block if block else "(no relevant memories)")
    finally:
        mem.close()
    return 0


def cmd_list(args) -> int:
    store = _open_store(args)
    try:
        records = store.list(
            user_id=args.user, memory_type=_mtype(args.type), limit=args.limit
        )
        if not records:
            print(f"No memories for user '{args.user}' in {_db_path(args)}")
            return 0
        for rec in records:
            _print_record(rec)
    finally:
        store.close()
    return 0


def cmd_stats(args) -> int:
    store = _open_store(args)
    try:
        print(f"database : {_db_path(args)}")
        print(f"user     : {args.user}")
        print(f"episodic : {store.count(user_id=args.user, memory_type=MemoryType.EPISODIC)}")
        print(f"semantic : {store.count(user_id=args.user, memory_type=MemoryType.SEMANTIC)}")
        print(f"total    : {store.count(user_id=args.user)} "
              f"(all users: {store.count()})")
    finally:
        store.close()
    return 0


def cmd_forget(args) -> int:
    store = _open_store(args)
    try:
        if store.delete(args.id):
            print(f"Forgot {args.id}")
            return 0
        print(f"No memory with id {args.id}")
        return 1
    finally:
        store.close()


def cmd_clear(args) -> int:
    store = _open_store(args)
    try:
        user = None if args.all_users else args.user
        scope = "ALL users" if args.all_users else f"user '{args.user}'"
        n = store.count(user_id=user)
        if n == 0:
            print(f"No memories for {scope}.")
            return 0
        if not args.yes and not _confirm(f"Delete {n} memories for {scope}?"):
            print("Aborted.")
            return 1
        print(f"Deleted {store.clear(user_id=user)} memories.")
    finally:
        store.close()
    return 0


def cmd_export(args) -> int:
    store = _open_store(args)
    try:
        user = None if args.all_users else args.user
        records = store.list(user_id=user, limit=1_000_000)
        payload = json.dumps([r.to_dict() for r in records],
                             ensure_ascii=False, indent=2)
    finally:
        store.close()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"Exported {len(records)} memories to {args.output}")
    else:
        print(payload)
    return 0


def cmd_import(args) -> int:
    with open(args.file, encoding="utf-8") as fh:
        payload = fh.read()
    if args.no_embed:
        store = _open_store(args)
        try:
            records = [MemoryRecord.from_dict(d) for d in json.loads(payload)]
            for rec in records:
                store.add(rec)
            print(f"Imported {len(records)} memories (without embeddings - "
                  f"keyword search only until re-embedded)")
        finally:
            store.close()
        return 0
    mem = _open_memory(args)
    try:
        count = mem.import_json(payload)
        print(f"Imported and re-embedded {count} memories")
    finally:
        mem.close()
    return 0


def cmd_prune(args) -> int:
    mem = _open_memory(args)  # prune is store-only: works without an API key
    try:
        deleted = mem.prune(
            max_age_days=args.days,
            max_importance=args.max_importance,
            user_id=args.user,
        )
        print(f"Pruned {deleted} stale episodic memories.")
    finally:
        mem.close()
    return 0


def cmd_ui(args) -> int:
    """Launch the local web dashboard."""
    from .ui.server import serve

    serve(
        db_path=_db_path(args),
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def cmd_chat(args) -> int:
    """Interactive REPL: a Gemini agent that remembers you across sessions."""
    from .core import MISSING_KEY_MESSAGE, MissingAPIKeyError
    from .middleware import MemoryMiddleware

    cfg = MemoryConfig.from_env()
    key = cfg.resolve_api_key()
    if key is None and not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
        raise MissingAPIKeyError(MISSING_KEY_MESSAGE)

    from google import genai

    client = genai.Client(api_key=key) if key else genai.Client()
    mem = _open_memory(args)
    middleware = MemoryMiddleware(mem, user_id=args.user,
                                  session_id=args.session)

    def chat_fn(messages: list[dict]) -> str:
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
            model=cfg.llm_model,
            contents=contents,
            config={"system_instruction": system or CHAT_SYSTEM},
        )
        return resp.text or ""

    chat = middleware.wrap(chat_fn)
    history: list[dict] = []
    print(f"Chatting as '{args.user}' "
          f"({mem.stats(user_id=args.user)['total']} memories on file). "
          f"Type 'quit' to exit.")
    try:
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            history.append({"role": "user", "content": user_input})
            reply = chat(list(history))
            history.append({"role": "assistant", "content": reply})
            print(f"\nagent> {reply}")
    finally:
        mem.close()
        print("\nMemories saved. See you next session!")
    return 0


# ---------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db", default=None,
        help="memory database path (default: memlayer.db or $MEMLAYER_DB_PATH)",
    )
    common.add_argument(
        "--user", default="default",
        help="user namespace (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        prog="memlayer",
        description="Persistent memory for LLM agents - powered by Gemini.",
        epilog="Run 'memlayer init' first, then try 'memlayer ui' or 'memlayer chat'.",
    )
    parser.add_argument("--version", action="version",
                        version=f"memlayer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("init", parents=[common],
                       help="one-time setup: save your API key and health-check")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", parents=[common],
                       help="check that everything is ready to use")
    p.add_argument("--live", action="store_true",
                   help="also make a tiny real API call")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("add", parents=[common],
                       help="store a memory (Gemini extracts durable facts)")
    p.add_argument("text", help="what to remember")
    p.add_argument("--session", default=None, help="session id tag")
    p.add_argument("--no-infer", action="store_true",
                   help="store verbatim, skip fact extraction")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("search", parents=[common], help="search memories")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=8)
    p.add_argument("--type", choices=["episodic", "semantic"], default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("context", parents=[common],
                       help="print the prompt block injected for a query")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=None, help="token budget")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("chat", parents=[common],
                       help="interactive agent that remembers you")
    p.add_argument("--session", default=None, help="session id tag")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ui", parents=[common],
                       help="open the web dashboard (chat + memory browser)")
    p.add_argument("--port", type=int, default=8765,
                   help="port to serve on (default: %(default)s)")
    p.add_argument("--no-browser", action="store_true",
                   help="don't open the browser automatically")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("list", parents=[common],
                       help="list stored memories (no API key needed)")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("--type", choices=["episodic", "semantic"], default=None)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("stats", parents=[common],
                       help="memory counts (no API key needed)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("forget", parents=[common],
                       help="delete one memory by id (no API key needed)")
    p.add_argument("id")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("clear", parents=[common],
                       help="delete all memories for a user (no API key needed)")
    p.add_argument("--all-users", action="store_true",
                   help="wipe every user's memories")
    p.add_argument("-y", "--yes", action="store_true",
                   help="skip the confirmation prompt")
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser("export", parents=[common],
                       help="export memories to JSON (no API key needed)")
    p.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    p.add_argument("--all-users", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", parents=[common],
                       help="import memories from a JSON export")
    p.add_argument("file")
    p.add_argument("--no-embed", action="store_true",
                   help="skip re-embedding (works without an API key)")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("prune", parents=[common],
                       help="drop stale, never-recalled episodic memories")
    p.add_argument("--days", type=float, default=90.0,
                   help="older than this many days (default: %(default)s)")
    p.add_argument("--max-importance", type=float, default=0.4)
    p.set_defaults(func=cmd_prune)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file()
    args = build_parser().parse_args(argv)
    from .core import MissingAPIKeyError

    try:
        return args.func(args) or 0
    except MissingAPIKeyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
