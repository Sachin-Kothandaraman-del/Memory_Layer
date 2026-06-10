# memlayer

**The transparent, self-reflecting memory layer for LLM agents** — episodic +
semantic memory with Gemini-powered extraction, consolidation, and hybrid
retrieval. One SQLite file, no external services.

Most agent memory is a black box: facts appear, facts vanish, nobody knows
why. memlayer is built around the opposite idea — **glass-box memory**:

- **Every fact has provenance** — click any memory and see the raw episodes
  it was distilled from, every version it went through, and the LLM's stated
  reasoning for each change (a permanent audit trail).
- **Memory never silently lies about the past** — updates *supersede* instead
  of overwrite. Ask `as_of=<timestamp>` and retrieval answers with what was
  believed *then* (time travel).
- **It forgets like a human** — an Ebbinghaus-style forgetting curve: every
  memory has a strength that grows when recalled and decays when ignored.
  Trivia fades in weeks; reinforced facts become near-permanent. Faded
  memories are excluded, not deleted — recoverable until pruned.
- **It reflects** — a sleep-style consolidation pass (`memlayer reflect`)
  reviews recent episodes and distills higher-order, evidence-cited insights
  ("user is under recurring deployment pressure") that per-message extraction
  can't see.
- **Local-first privacy** — your agent's memory is a file you own. Optional
  PII redaction runs *before* anything reaches the API, and "off the record"
  requests are honored (nothing stored, audit entry only).

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

## Core features

- **Two memory types** — *episodic* (raw events: what was said, when, in which
  session) and *semantic* (durable facts distilled by Gemini: identity,
  preferences, goals, constraints, relationships, reflected insights...)
- **LLM consolidation with an audit trail** — new facts are reconciled against
  existing ones (ADD / UPDATE / DELETE / NONE); every decision is recorded
  with the LLM's reasoning, and updates keep the old version as history
- **Hybrid retrieval** — Gemini embeddings (cosine) + SQLite FTS5 (BM25), fused
  with reciprocal-rank fusion, re-ranked by
  `similarity·w₁ + retention·w₂ + importance·w₃`, de-duplicated with MMR
- **Forgetting curve** — `retention = 0.5^(hours_since_recall / half_life)`
  where the half-life scales with strength (grows ×1.8 per recall, capped) and
  importance; below the retention floor a memory has faded out of retrieval
- **Time travel** — `search(..., as_of=ts)` and `history(id)` expose the full
  belief timeline; nothing is destroyed by consolidation
- **Reflection** — `reflect()` reviews recent episodes and writes
  evidence-cited insights, integrated through the same consolidation pipeline
- **Privacy guards** — `redact_pii=True` scrubs emails/phones/cards/SSNs/IPs
  with pure regex *before* embedding or extraction; "off the record" phrases
  skip storage entirely; `forget()` is a true hard delete
- **Token-budgeted context** — `get_context()` packs the best memories into a
  prompt block that fits your budget
- **Drop-in middleware** — wrap any `chat_fn(messages) -> str`; recall happens
  before inference, recording happens after, on a background thread (zero
  added latency on the write path)
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
  *"recalled N memories"* note with each memory's score breakdown
  (similarity · retention · importance · strength) — click one to open its
  full story
- **Memories** — live hybrid search, type filter, importance + retention
  bars, recall counts, one-click forget, a **show history** toggle that
  reveals superseded versions and faded memories, and a **✨ Reflect** button
- **Memory drawer** — click any memory: its belief timeline (every version,
  when it held), the source episodes it was derived from, and its audit trail
  with the LLM's reasoning
- **Add memory** — store something manually and see which facts Gemini
  extracted, what the consolidator decided and *why*, and any PII redactions
- **Activity** — the glass-box audit log: every ADD / UPDATE / RETRACT /
  REFLECT / REDACT with reasoning

Works without an API key too (browse/manage mode with keyword search).
Use `--port` to change the port, `--no-browser` to skip auto-opening.

## Command line

You don't have to write any code to use the memory layer:

```bash
memlayer add "I'm Priya, I lead the data platform team" --user priya
memlayer search "who leads data platform?" --user priya
memlayer context "schedule a sync" --user priya   # the block an agent would see
memlayer chat --user priya                        # interactive remembering agent
memlayer reflect --user priya                     # distill insights from recent episodes
memlayer history <id>                             # one memory's versions, sources, audit
memlayer audit                                    # the glass-box activity log
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
| `search(..., as_of=ts)` | time travel: what was believed at that moment |
| `reflect(user_id=)` | distill evidence-cited insights from recent episodes |
| `history(id)` | belief timeline + source episodes + audit trail |
| `audit_log(user_id=)` | recent memory operations with LLM reasoning |
| `forget(id)` / `clear(user_id=)` | hard-delete one / all memories (audited) |
| `prune(max_age_days=, min_retention=)` | drop stale or fully-faded episodes |
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

- **Write path**: every `add()` first passes the privacy guards
  (never-remember check, optional PII redaction), stores the raw episode,
  then (if `infer`) Gemini extracts candidate facts. Each fact is embedded
  and compared to existing semantic memories; if nothing is ≥
  `consolidation_sim_threshold` it is added directly (no LLM call), otherwise
  Gemini decides ADD/UPDATE/DELETE/NONE with stated reasoning. UPDATE writes
  a *new version* and stamps the old one with `valid_until`/`superseded_by`;
  DELETE retracts (invalidates) rather than destroys. Every decision lands in
  the audit log. Extraction failures never lose the episode.
- **Read path**: vector and keyword candidates are fused by reciprocal-rank
  fusion (cosine is kept as the similarity signal), scored with
  forgetting-curve retention and stored importance, floor-filtered (faded
  memories drop out), then MMR-filtered so near-duplicates don't waste the
  token budget. Recalled memories are reinforced: access stats bumped and
  strength multiplied, so they decay slower next time. `as_of` queries are
  read-only (no reinforcement) and see the world as it was believed then.
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
