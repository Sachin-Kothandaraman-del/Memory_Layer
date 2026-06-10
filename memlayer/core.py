"""MemoryLayer: the main orchestrator tying storage, extraction, retrieval
and consolidation together behind a small API.

    mem = MemoryLayer(api_key="...")                     # or from_env()
    mem.add("I'm Priya, I lead the data platform team", user_id="u1")
    hits = mem.search("who runs data platform?", user_id="u1")
    block = mem.get_context("plan the migration", user_id="u1")
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Sequence

from .config import MemoryConfig
from .consolidation import ConsolidationResult, Consolidator
from .embeddings import Embedder, GeminiEmbedder
from .extraction import FactExtractor
from .llm import LLM, GeminiLLM
from .models import ExtractedFact, MemoryRecord, MemoryType, ScoredMemory
from .retrieval import Retriever
from .storage.base import MemoryStore
from .storage.sqlite_store import SQLiteMemoryStore

logger = logging.getLogger("memlayer")

MISSING_KEY_MESSAGE = """\
No Gemini API key found.

Set one up (any of these):
  1. PowerShell:   $env:GEMINI_API_KEY = "your-key"     (current session)
                   setx GEMINI_API_KEY "your-key"       (persistent)
  2. Create a .env file next to your script containing:
                   GEMINI_API_KEY=your-key
     (or simply run:  memlayer init)
  3. In code:      MemoryLayer(api_key="your-key")

Get a free key at https://aistudio.google.com/apikey"""


class MissingAPIKeyError(RuntimeError):
    """Raised when an operation needs the Gemini API but no key is configured."""


class MemoryLayer:
    """Persistent episodic + semantic memory for any LLM agent."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        api_key: str | None = None,
        db_path: str | None = None,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ):
        self.config = config or MemoryConfig()
        if api_key is not None:
            self.config.api_key = api_key
        if db_path is not None:
            self.config.db_path = db_path

        self.store = store or SQLiteMemoryStore(self.config.db_path)

        # Gemini clients are created lazily so store-only operations
        # (list/stats/forget/clear/export/prune) work without an API key.
        self._embedder = embedder
        self._llm = llm
        self._extractor: FactExtractor | None = None
        self._consolidator: Consolidator | None = None
        self._retriever: Retriever | None = None

        # single worker => writes for one process are applied in order
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memlayer-writer"
        )
        self._pending: list[Future] = []

    def _require_key(self) -> str | None:
        key = self.config.resolve_api_key()
        if key is None and not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
            raise MissingAPIKeyError(MISSING_KEY_MESSAGE)
        return key

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder(
                api_key=self._require_key(),
                model=self.config.embed_model,
                dim=self.config.embed_dim,
                batch_size=self.config.embed_batch_size,
                max_retries=self.config.max_retries,
            )
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = GeminiLLM(
                api_key=self._require_key(),
                model=self.config.llm_model,
                max_retries=self.config.max_retries,
            )
        return self._llm

    @property
    def extractor(self) -> FactExtractor:
        if self._extractor is None:
            self._extractor = FactExtractor(self.llm)
        return self._extractor

    @property
    def consolidator(self) -> Consolidator:
        if self._consolidator is None:
            self._consolidator = Consolidator(self.llm)
        return self._consolidator

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, self.embedder, self.config)
        return self._retriever

    @classmethod
    def from_env(cls, **overrides) -> "MemoryLayer":
        return cls(config=MemoryConfig.from_env(**overrides))

    # ------------------------------------------------------------------ write

    def add(
        self,
        content: str,
        *,
        user_id: str = "default",
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        infer: bool | None = None,
        wait: bool = True,
    ) -> dict[str, Any] | Future:
        """Store an event as episodic memory and (optionally) extract semantic
        facts from it.

        With ``wait=False`` the write runs on a background thread so the
        agent's response latency is unaffected; call :meth:`flush` to drain.
        Returns a summary dict (or a Future of one when ``wait=False``).
        """
        if not content or not content.strip():
            return {"episodic": None, "facts": []}
        kwargs = dict(
            user_id=user_id, agent_id=agent_id, session_id=session_id,
            metadata=metadata or {}, importance=importance, infer=infer,
        )
        if wait:
            return self._write(content, **kwargs)
        self._reap_pending()
        future = self._executor.submit(self._write, content, **kwargs)
        self._pending.append(future)
        return future

    def add_messages(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any] | Future:
        """Store a conversation turn given OpenAI-style message dicts
        (``{"role": ..., "content": ...}``)."""
        transcript = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        return self.add(transcript, **kwargs)

    def _write(
        self,
        content: str,
        user_id: str,
        agent_id: str | None,
        session_id: str | None,
        metadata: dict[str, Any],
        importance: float | None,
        infer: bool | None,
    ) -> dict[str, Any]:
        episodic = MemoryRecord(
            content=content,
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            importance=(
                importance
                if importance is not None
                else self.config.episodic_default_importance
            ),
            embedding=self.embedder.embed_documents([content])[0],
        )
        self.store.add(episodic)

        facts: list[dict[str, Any]] = []
        should_infer = self.config.extract_semantic if infer is None else infer
        if should_infer and len(content.strip()) >= self.config.min_extraction_chars:
            for fact in self.extractor.extract(content):
                result = self._integrate_fact(
                    fact, user_id=user_id, agent_id=agent_id,
                    source_id=episodic.id,
                )
                facts.append(
                    {
                        "content": fact.content,
                        "added": [r.id for r in result.added],
                        "updated": [r.id for r in result.updated],
                        "deleted": result.deleted,
                        "skipped": result.skipped,
                    }
                )
        return {"episodic": episodic.id, "facts": facts}

    def _integrate_fact(
        self,
        fact: ExtractedFact,
        user_id: str,
        agent_id: str | None,
        source_id: str,
    ) -> ConsolidationResult:
        """Consolidate one extracted fact into the semantic store."""
        embedding = self.embedder.embed_documents([fact.content])[0]
        similar = self.store.vector_search(
            embedding,
            limit=self.config.consolidation_top_k,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
        )

        def make_record(content: str) -> MemoryRecord:
            return MemoryRecord(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                user_id=user_id,
                agent_id=agent_id,
                importance=fact.importance,
                category=fact.category,
                source_ids=[source_id],
                embedding=(
                    embedding
                    if content == fact.content
                    else self.embedder.embed_documents([content])[0]
                ),
            )

        close_enough = [
            (rec, sim)
            for rec, sim in similar
            if sim >= self.config.consolidation_sim_threshold
        ]
        if not self.config.consolidate or not close_enough:
            record = make_record(fact.content)
            self.store.add(record)
            return ConsolidationResult(added=[record], updated=[], deleted=[])

        result = ConsolidationResult(added=[], updated=[], deleted=[])
        for op in self.consolidator.decide(fact.content, close_enough):
            kind = (op.get("op") or "").upper()
            if kind == "ADD":
                record = make_record(fact.content)
                self.store.add(record)
                result.added.append(record)
            elif kind == "UPDATE" and op.get("id"):
                existing = self.store.get(op["id"])
                if existing is None:
                    continue
                new_content = (op.get("content") or fact.content).strip()
                existing.content = new_content
                existing.embedding = self.embedder.embed_documents([new_content])[0]
                existing.importance = max(existing.importance, fact.importance)
                existing.source_ids = list({*existing.source_ids, source_id})
                self.store.update(existing)
                result.updated.append(existing)
            elif kind == "DELETE" and op.get("id"):
                if self.store.delete(op["id"]):
                    result.deleted.append(op["id"])
            elif kind == "NONE":
                result.skipped = True
        if not (result.added or result.updated or result.deleted):
            result.skipped = True
        return result

    # ------------------------------------------------------------------- read

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        reinforce: bool = True,
    ) -> list[ScoredMemory]:
        """Hybrid search over stored memories, best-scored first."""
        return self.retriever.search(
            query,
            limit=limit,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            reinforce=reinforce,
        )

    def get_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        token_budget: int | None = None,
        limit: int = 20,
    ) -> str:
        """Build a memory block to inject into a prompt, packed to fit
        ``token_budget`` (estimated). Returns "" when nothing relevant."""
        budget = token_budget or self.config.default_token_budget
        results = self.search(
            query, limit=limit, user_id=user_id, agent_id=agent_id
        )
        if not results:
            return ""

        facts = [s for s in results if s.record.memory_type == MemoryType.SEMANTIC]
        events = [s for s in results if s.record.memory_type == MemoryType.EPISODIC]

        header = (
            "Long-term memory (recalled for this request — may be relevant "
            "context about the user and past sessions):"
        )
        used = self._estimate_tokens(header)
        lines: list[str] = [header]

        def push(section: str, items: list[ScoredMemory]) -> None:
            nonlocal used
            if not items:
                return
            section_line = f"\n{section}"
            section_cost = self._estimate_tokens(section_line)
            opened = False
            for s in items:
                date = _dt.datetime.fromtimestamp(
                    s.record.updated_at
                ).strftime("%Y-%m-%d")
                line = f"- ({date}) {s.record.content}"
                cost = self._estimate_tokens(line)
                extra = section_cost if not opened else 0
                if used + cost + extra > budget:
                    continue
                if not opened:
                    lines.append(section_line)
                    used += section_cost
                    opened = True
                lines.append(line)
                used += cost

        push("Known facts:", facts)
        push("Past events:", events)
        return "\n".join(lines) if len(lines) > 1 else ""

    # -------------------------------------------------------------- management

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def forget(self, memory_id: str) -> bool:
        """Delete a single memory."""
        return self.store.delete(memory_id)

    def clear(self, user_id: str | None = None) -> int:
        """Delete all memories (optionally scoped to one user)."""
        return self.store.clear(user_id=user_id)

    def prune(
        self,
        *,
        max_age_days: float = 90.0,
        max_importance: float = 0.4,
        max_access_count: int = 0,
        user_id: str | None = None,
    ) -> int:
        """Forget old, unimportant, never-recalled episodic memories."""
        cutoff = time.time() - max_age_days * 86400.0
        deleted = 0
        for rec in self.store.list(
            user_id=user_id, memory_type=MemoryType.EPISODIC, limit=100_000
        ):
            if (
                rec.updated_at < cutoff
                and rec.importance <= max_importance
                and rec.access_count <= max_access_count
            ):
                if self.store.delete(rec.id):
                    deleted += 1
        return deleted

    def summarize_session(
        self, session_id: str, user_id: str = "default", delete_episodic: bool = False
    ) -> MemoryRecord | None:
        """Compress a session's episodic trace into one semantic summary."""
        episodes = self.store.list(
            user_id=user_id, session_id=session_id,
            memory_type=MemoryType.EPISODIC, limit=500,
        )
        if not episodes:
            return None
        episodes.sort(key=lambda r: r.created_at)
        transcript = "\n\n".join(e.content for e in episodes)
        summary = self.llm.generate(
            "Summarize this session into a short paragraph capturing decisions, "
            "outcomes, and anything worth remembering long-term:\n\n" + transcript
        ).strip()
        if not summary:
            return None
        record = MemoryRecord(
            content=f"Session summary: {summary}",
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            session_id=session_id,
            category="event",
            importance=0.6,
            source_ids=[e.id for e in episodes],
            embedding=self.embedder.embed_documents([summary])[0],
        )
        self.store.add(record)
        if delete_episodic:
            for e in episodes:
                self.store.delete(e.id)
        return record

    def export(self, user_id: str | None = None) -> str:
        """Dump memories as a JSON string (embeddings excluded)."""
        records = self.store.list(user_id=user_id, limit=1_000_000)
        return json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2)

    def import_json(self, payload: str, re_embed: bool = True) -> int:
        """Load memories from an :meth:`export` dump."""
        records = [MemoryRecord.from_dict(d) for d in json.loads(payload)]
        if re_embed and records:
            vectors = self.embedder.embed_documents([r.content for r in records])
            for rec, vec in zip(records, vectors):
                rec.embedding = vec
        for rec in records:
            self.store.add(rec)
        return len(records)

    def stats(self, user_id: str | None = None) -> dict[str, int]:
        return {
            "total": self.store.count(user_id=user_id),
            "episodic": self.store.count(user_id=user_id,
                                         memory_type=MemoryType.EPISODIC),
            "semantic": self.store.count(user_id=user_id,
                                         memory_type=MemoryType.SEMANTIC),
        }

    # -------------------------------------------------------------- lifecycle

    def flush(self, timeout: float | None = None) -> None:
        """Wait for queued background writes to land."""
        pending, self._pending = self._pending, []
        for future in pending:
            try:
                future.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.error("background memory write failed: %s", exc)

    def _reap_pending(self) -> None:
        """Drop finished background writes, surfacing any errors."""
        still_running: list[Future] = []
        for future in self._pending:
            if future.done():
                exc = future.exception()
                if exc is not None:
                    logger.error("background memory write failed: %s", exc)
            else:
                still_running.append(future)
        self._pending = still_running

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)
        self.store.close()

    def __enter__(self) -> "MemoryLayer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.config.chars_per_token) + 1
