# memlayer

**Persistent memory middleware for any LLM agent** — episodic + semantic memory with
Gemini-powered extraction, consolidation, and hybrid retrieval. One SQLite file, no
external services.

```
                       ┌──────────────────────────────────────────────┐
   your agent          │                  memlayer                    │
┌─────────────┐ before │  ┌────────────┐   hybrid search   ┌────────┐ │
│  messages ──┼────────┼─▶│ Retriever  │◀─────────────────▶│ SQLite │ │
│             │        │  │ embed+FTS5 │  RRF · recency ·  │ store  │ │
│  LLM call   │        │  │ rerank+MMR │  importance · MMR │vec+FTS5│ │
│             │ after  │  └────────────┘                   └────────┘ │
│  response ──┼────────┼─▶┌────────────┐  ┌──────────────┐     ▲      │
└─────────────┘        │  │ Extractor  │─▶│ Consolidator │─────┘      │
                       │  │  (Gemini)  │  │ ADD/UPDATE/  │ semantic   │
                       │  └────────────┘  │ DELETE/NONE  │ facts      │
                       │   episodic ──────└──────────────┘            │
                       └──────────────────────────────────────────────┘
```

## Features

- **Two memory types** — *episodic* (raw events: what was said, when, in which
  session) and *semantic* (durable facts distilled by Gemini: identity,
  preferences, goals, constraints, relationships...)
- **LLM consolidation** — new facts are reconciled against existing ones
  (ADD / UPDATE / DELETE / NONE), so the store converges to current truth
  instead of accumulating contradictions
- **Hybrid retrieval** — Gemini embeddings (cosine) + SQLite FTS5 (BM25), fused
  with reciprocal-rank fusion, re-ranked by
  `similarity·w₁ + recency-decay·w₂ + importance·w₃`, de-duplicated with MMR
- **Token-budgeted context** — `get_context()` packs the best memories into a
  prompt block that fits your budget
- **Drop-in middleware** — wrap any `chat_fn(messages) -> str`; recall happens
  before inference, recording happens after, on a background thread (zero
  added latency on the write path)
- **Memory hygiene** — reinforcement on recall, `prune()` for stale low-value
  episodes, `summarize_session()` to compress a session into one fact,
  export/import as JSON
- **Multi-tenant** — namespaced by `user_id` / `agent_id` / `session_id`
- **Pluggable** — swap the store (subclass `MemoryStore`), the embedder, or the
  LLM; defaults are Gemini + SQLite

## Setup (2 minutes)

```bash
pip install -e .            # from this directory
memlayer init               # paste your API key once — saved to .env, then health-checked
memlayer ui                 # open the web dashboard (chat + memory browser)
```

`memlayer init` walks you through getting a free key
(https://aistudio.google.com/apikey). Alternatively set `GEMINI_API_KEY`
yourself — environment variable, `.env` file (auto-loaded, no python-dotenv
needed), or `MemoryLayer(api_key=...)` in code. If anything is off, any
command that needs the API prints exactly how to fix it, and
`memlayer doctor --live` verifies your key with a real API call.

## Web dashboard

`memlayer ui` starts a local dashboard at http://127.0.0.1:8765 (stdlib HTTP
server — no extra dependencies, bound to localhost only):

- **Chat** — talk to a Gemini agent that remembers you; every reply shows a
  collapsible *"recalled N memories"* note with the exact context that was
  injected, so you can watch the memory work
- **Memories** — live hybrid search, filter by type, importance bars, recall
  counts, one-click forget, per-user clearing
- **Add memory** — store something manually and see which facts Gemini
  extracted and whether they were added, updated, or already known

Works without an API key too (browse/manage mode with keyword search).
Use `--port` to change the port, `--no-browser` to skip auto-opening.

## Command line

You don't have to write any code to use the memory layer:

```bash
memlayer add "I'm Priya, I lead the data platform team" --user priya
memlayer search "who leads data platform?" --user priya
memlayer context "schedule a sync" --user priya   # the block an agent would see
memlayer chat --user priya                        # interactive remembering agent
memlayer list / stats / forget <id> / clear       # inspect & manage (no API key needed)
memlayer export -o backup.json / import backup.json
memlayer prune --days 90                          # drop stale never-recalled episodes
memlayer doctor --live                            # end-to-end health check
```

Every command takes `--db path` and `--user name`; store-management commands
work entirely offline.

## Quickstart

```python
from memlayer import MemoryLayer

mem = MemoryLayer.from_env(db_path="memories.db")

# write: stores the episode AND extracts semantic facts with Gemini
mem.add("user: I'm Priya, I lead the data platform team. Never schedule "
        "meetings on Fridays.", user_id="priya")

# read: hybrid search, scored and de-duplicated
for hit in mem.search("when can I meet her?", user_id="priya"):
    print(hit.score, hit.record.content)

# or get a ready-to-inject prompt block, packed to a token budget
context = mem.get_context("schedule a sync", user_id="priya", token_budget=800)
```

## Drop-in middleware (any agent, any framework)

```python
from memlayer import MemoryLayer, MemoryMiddleware

mem = MemoryLayer.from_env(db_path="memories.db")
mw = MemoryMiddleware(mem, user_id="priya")

chat = mw.wrap(my_chat_fn)   # my_chat_fn(messages: list[dict]) -> str
reply = chat([{"role": "user", "content": "plan my week"}])
# 1. relevant memories injected into the system message
# 2. my_chat_fn called with the augmented messages
# 3. the exchange recorded + facts extracted in the background
```

Or control both phases yourself (works with streaming, tool loops, anything):

```python
messages = mw.before(messages)   # inject recalled context
response = run_inference(messages)
mw.after(messages, response)     # record (non-blocking)
```

Decorator form:

```python
from memlayer import with_memory

@with_memory(mem, user_id="priya")
def chat(messages): ...
```

## API overview

| Method | What it does |
|---|---|
| `add(text, user_id=, session_id=, infer=, wait=)` | store episode, extract + consolidate facts; `wait=False` → background |
| `add_messages([{role, content}, ...])` | same, from OpenAI-style messages |
| `search(query, limit=, memory_type=, ...)` | hybrid retrieval → `list[ScoredMemory]` |
| `get_context(query, token_budget=)` | formatted prompt block (or `""`) |
| `forget(id)` / `clear(user_id=)` | delete one / all memories |
| `prune(max_age_days=, max_importance=)` | drop stale, never-recalled episodes |
| `summarize_session(session_id)` | compress a session into one semantic memory |
| `export(user_id=)` / `import_json(payload)` | JSON backup / restore |
| `stats(user_id=)` | counts by memory type |
| `flush()` / `close()` | drain background writes / shut down |

## Configuration

Everything is tunable via `MemoryConfig`:

```python
from memlayer import MemoryConfig, MemoryLayer

mem = MemoryLayer(MemoryConfig(
    db_path="memories.db",
    llm_model="gemini-2.5-flash",
    embed_model="gemini-embedding-001",
    embed_dim=768,
    weight_similarity=0.65,        # retrieval score weights
    weight_recency=0.15,
    weight_importance=0.20,
    recency_half_life_hours=168,   # one week
    mmr_lambda=0.7,                # relevance vs. diversity
    default_token_budget=1200,
    consolidation_sim_threshold=0.75,
))
```

## Design notes

- **Write path**: every `add()` stores the raw episode immediately, then (if
  `infer`) Gemini extracts candidate facts. Each fact is embedded and compared
  to existing semantic memories; if nothing is ≥ `consolidation_sim_threshold`
  it is added directly (no LLM call), otherwise Gemini decides
  ADD/UPDATE/DELETE/NONE. Extraction failures never lose the episode.
- **Read path**: vector and keyword candidates are fused by reciprocal-rank
  fusion (cosine is kept as the similarity signal), scored with exponential
  recency decay (configurable half-life) and stored importance, then filtered
  by MMR so near-duplicate memories don't waste the token budget. Recalled
  memories are "reinforced" (access stats bumped) for use by `prune()`.
- **Embeddings** are L2-normalized at write time (truncated Gemini embeddings
  are not pre-normalized), so similarity is a dot product.
- **Concurrency**: one background writer thread keeps writes ordered;
  SQLite access is serialized behind a lock; WAL mode keeps readers fast.

## Run the demos

```bash
python examples/quickstart.py    # end-to-end write/extract/search/context
python examples/demo_agent.py    # interactive CLI agent that remembers you
```

## Tests

No API key needed — the suite uses deterministic fakes:

```bash
pytest
```
# Memory_Layer
