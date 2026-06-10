"""MemoryLayer: the main orchestrator tying storage, extraction, retrieval,
consolidation, reflection, and privacy together behind a small API.

    mem = MemoryLayer(api_key="...")                     # or from_env()
    mem.add("I'm Priya, I lead the data platform team", user_id="u1")
    hits = mem.search("who runs data platform?", user_id="u1")
    block = mem.get_context("plan the migration", user_id="u1")
    mem.reflect(user_id="u1")        # distill higher-order insights
    mem.history(memory_id)           # provenance + audit trail of one memory
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
from .models import (
    AuditEntry,
    ExtractedFact,
    MemoryRecord,
    MemoryType,
    ScoredMemory,
)
from .privacy import matches_never_remember, redact_pii
from .reflection import Reflector
from .retrieval import Retriever, retention_of
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
        # (list/stats/forget/clear/export/prune/history) work without a key.
        self._embedder = embedder
        self._llm = llm
        self._extractor: FactExtractor | None = None
        self._consolidator: Consolidator | None = None
        self._retriever: Retriever | None = None
        self._reflector: Reflector | None = None

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

    @property
    def reflector(self) -> Reflector:
        if self._reflector is None:
            self._reflector = Reflector(self.llm)
        return self._reflector

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
        # privacy guard 1: explicit "don't remember this" requests are honored
        if matches_never_remember(content, self.config.never_remember_patterns):
            self._audit("SKIPPED_PRIVATE", user_id,
                        reasoning="input matched a never-remember pattern")
            return {"episodic": None, "facts": [], "skipped_private": True}

        # privacy guard 2: optional PII redaction BEFORE embedding/extraction,
        # so PII never reaches the API and is never stored
        redactions: dict[str, int] = {}
        if self.config.redact_pii:
            content, redactions = redact_pii(content)
            if redactions:
                self._audit("REDACT", user_id, detail={"redacted": redactions})

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
                    source_ids=[episodic.id],
                )
                facts.append(
                    {
                        "content": fact.content,
                        "added": [r.id for r in result.added],
                        "updated": [r.id for r in result.updated],
                        "deleted": result.deleted,
                        "skipped": result.skipped,
                        "operations": result.operations,
                    }
                )
        out: dict[str, Any] = {"episodic": episodic.id, "facts": facts}
        if redactions:
            out["redacted"] = redactions
        return out

    def _integrate_fact(
        self,
        fact: ExtractedFact,
        user_id: str,
        agent_id: str | None,
        source_ids: list[str],
    ) -> ConsolidationResult:
        """Consolidate one fact into the semantic store.

        With ``keep_history`` (default), updates and deletes never destroy:
        the old version gets ``valid_until``/``superseded_by`` set and a new
        version is written, so the full belief timeline stays queryable.
        """
        embedding = self.embedder.embed_documents([fact.content])[0]
        similar = self.store.vector_search(
            embedding,
            limit=self.config.consolidation_top_k,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
        )

        def make_record(content: str, **overrides) -> MemoryRecord:
            return MemoryRecord(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                user_id=user_id,
                agent_id=agent_id,
                importance=overrides.pop("importance", fact.importance),
                category=overrides.pop("category", fact.category),
                source_ids=overrides.pop("source_ids", list(source_ids)),
                embedding=(
                    embedding
                    if content == fact.content
                    else self.embedder.embed_documents([content])[0]
                ),
                **overrides,
            )

        result = ConsolidationResult(added=[], updated=[], deleted=[])

        close_enough = [
            (rec, sim)
            for rec, sim in similar
            if sim >= self.config.consolidation_sim_threshold
        ]
        if not self.config.consolidate or not close_enough:
            record = make_record(fact.content)
            self.store.add(record)
            reasoning = "no similar existing memory; stored as new fact"
            self._audit("ADD", user_id, record.id, reasoning=reasoning,
                        detail={"content": record.content,
                                "sources": source_ids})
            result.added.append(record)
            result.operations.append({"op": "ADD", "reasoning": reasoning})
            return result

        now = time.time()
        for op in self.consolidator.decide(fact.content, close_enough):
            kind = (op.get("op") or "").upper()
            reasoning = (op.get("reasoning") or "").strip() or None
            if kind == "ADD":
                record = make_record(fact.content)
                self.store.add(record)
                self._audit("ADD", user_id, record.id, reasoning=reasoning,
                            detail={"content": record.content,
                                    "sources": source_ids})
                result.added.append(record)
            elif kind == "UPDATE" and op.get("id"):
                old = self.store.get(op["id"])
                if old is None:
                    continue
                new_content = (op.get("content") or fact.content).strip()
                new = make_record(
                    new_content,
                    importance=max(old.importance, fact.importance),
                    category=old.category or fact.category,
                    source_ids=sorted({*old.source_ids, *source_ids}),
                    strength=old.strength,   # supersession keeps reinforcement
                    valid_from=now,
                )
                self.store.add(new)
                if self.config.keep_history:
                    old.valid_until = now
                    old.superseded_by = new.id
                    self.store.update(old)
                else:
                    self.store.delete(old.id)
                self._audit("UPDATE", user_id, new.id, reasoning=reasoning,
                            detail={"old_id": old.id,
                                    "old_content": old.content,
                                    "new_content": new_content})
                result.updated.append(new)
            elif kind == "DELETE" and op.get("id"):
                old = self.store.get(op["id"])
                if old is None:
                    continue
                if self.config.keep_history:
                    old.valid_until = now
                    self.store.update(old)
                else:
                    self.store.delete(old.id)
                self._audit("RETRACT", user_id, old.id, reasoning=reasoning,
                            detail={"content": old.content})
                result.deleted.append(old.id)
            elif kind == "NONE":
                self._audit("NONE", user_id, reasoning=reasoning,
                            detail={"content": fact.content})
                result.skipped = True
            if kind in ("ADD", "UPDATE", "DELETE", "NONE"):
                result.operations.append({"op": kind, "reasoning": reasoning})
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
        include_faded: bool = False,
        as_of: float | None = None,
    ) -> list[ScoredMemory]:
        """Hybrid search over stored memories, best-scored first.

        ``as_of`` (epoch seconds) time-travels: results reflect what was
        believed at that moment, including since-superseded facts.
        ``include_faded=True`` bypasses the forgetting curve.
        """
        return self.retriever.search(
            query,
            limit=limit,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            reinforce=reinforce,
            include_faded=include_faded,
            as_of=as_of,
        )

    def build_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        token_budget: int | None = None,
        limit: int = 20,
        as_of: float | None = None,
    ) -> tuple[str, list[ScoredMemory]]:
        """Like :meth:`get_context`, but also returns which memories were
        packed into the block (for glass-box display)."""
        budget = token_budget or self.config.default_token_budget
        results = self.search(
            query, limit=limit, user_id=user_id, agent_id=agent_id, as_of=as_of
        )
        if not results:
            return "", []

        facts = [s for s in results if s.record.memory_type == MemoryType.SEMANTIC]
        events = [s for s in results if s.record.memory_type == MemoryType.EPISODIC]

        header = (
            "Long-term memory (recalled for this request — may be relevant "
            "context about the user and past sessions):"
        )
        used = self._estimate_tokens(header)
        lines: list[str] = [header]
        included: list[ScoredMemory] = []

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
                included.append(s)

        push("Known facts:", facts)
        push("Past events:", events)
        if len(lines) <= 1:
            return "", []
        return "\n".join(lines), included

    def get_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        token_budget: int | None = None,
        limit: int = 20,
        as_of: float | None = None,
    ) -> str:
        """Build a memory block to inject into a prompt, packed to fit
        ``token_budget`` (estimated). Returns "" when nothing relevant."""
        return self.build_context(
            query, user_id=user_id, agent_id=agent_id,
            token_budget=token_budget, limit=limit, as_of=as_of,
        )[0]

    # ---------------------------------------------------------------- insight

    def reflect(
        self,
        user_id: str = "default",
        *,
        agent_id: str | None = None,
        window: int | None = None,
    ) -> dict[str, Any]:
        """Sleep-style consolidation: review recent episodes and distill
        higher-order insights, each citing its evidence episodes."""
        window = window or self.config.reflection_window
        episodes = self.store.list(
            user_id=user_id, memory_type=MemoryType.EPISODIC, limit=window
        )
        if not episodes:
            return {"insights": [], "examined_episodes": 0}
        known = self.store.list(
            user_id=user_id, memory_type=MemoryType.SEMANTIC, limit=200
        )

        created: list[MemoryRecord] = []
        for insight in self.reflector.reflect(episodes, known):
            fact = ExtractedFact(
                content=insight.content,
                category="insight",
                importance=round(0.5 + 0.4 * insight.confidence, 2),
            )
            result = self._integrate_fact(
                fact, user_id=user_id, agent_id=agent_id,
                source_ids=insight.evidence,
            )
            for rec in result.added + result.updated:
                self._audit(
                    "REFLECT", user_id, rec.id,
                    reasoning=insight.reasoning or None,
                    detail={"insight": insight.content,
                            "confidence": insight.confidence,
                            "evidence": insight.evidence},
                )
                created.append(rec)
        return {
            "insights": [r.to_dict() for r in created],
            "examined_episodes": len(episodes),
        }

    # ------------------------------------------------------------ glass-box

    def history(self, memory_id: str) -> dict[str, Any] | None:
        """Full provenance of one memory: version chain (oldest→newest),
        source episodes, and its audit trail."""
        record = self.store.get(memory_id)
        if record is None:
            return None

        # walk back through superseded versions
        chain = [record]
        cursor = record
        while True:
            prev = self.store.predecessor(cursor.id)
            if prev is None or any(prev.id == c.id for c in chain):
                break
            chain.append(prev)
            cursor = prev
        chain.reverse()  # oldest first
        # walk forward if this record itself was superseded
        cursor = record
        while cursor.superseded_by:
            nxt = self.store.get(cursor.superseded_by)
            if nxt is None or any(nxt.id == c.id for c in chain):
                break
            chain.append(nxt)
            cursor = nxt

        sources = []
        for sid in record.source_ids:
            src = self.store.get(sid)
            if src is not None:
                sources.append(src.to_dict())

        audit = [
            e.to_dict()
            for e in self.store.get_audit(limit=200)
            if e.memory_id in {c.id for c in chain}
        ]
        return {
            "record": record.to_dict(),
            "retention": round(retention_of(record, self.config), 4),
            "versions": [c.to_dict() for c in chain],
            "sources": sources,
            "audit": audit,
        }

    def audit_log(self, user_id: str | None = None, limit: int = 100) -> list[dict]:
        """Recent glass-box audit entries, newest first."""
        return [e.to_dict() for e in
                self.store.get_audit(user_id=user_id, limit=limit)]

    def retention(self, record: MemoryRecord) -> float:
        """Current forgetting-curve retention (0..1) for a record."""
        return retention_of(record, self.config)

    # -------------------------------------------------------------- management

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def forget(self, memory_id: str) -> bool:
        """Hard-delete a single memory (privacy: truly gone, audit kept)."""
        record = self.store.get(memory_id)
        deleted = self.store.delete(memory_id)
        if deleted:
            self._audit(
                "FORGET",
                record.user_id if record else None,
                memory_id,
                reasoning="explicit user deletion",
            )
        return deleted

    def clear(self, user_id: str | None = None) -> int:
        """Delete all memories (optionally scoped to one user)."""
        deleted = self.store.clear(user_id=user_id)
        if deleted:
            self._audit("CLEAR", user_id, detail={"deleted": deleted})
        return deleted

    def prune(
        self,
        *,
        max_age_days: float = 90.0,
        max_importance: float = 0.4,
        max_access_count: int = 0,
        min_retention: float | None = None,
        user_id: str | None = None,
    ) -> int:
        """Forget old, unimportant, never-recalled episodic memories.
        With ``min_retention``, also drops episodes that have faded below
        that retention level regardless of age."""
        cutoff = time.time() - max_age_days * 86400.0
        deleted = 0
        for rec in self.store.list(
            user_id=user_id, memory_type=MemoryType.EPISODIC, limit=100_000
        ):
            stale = (
                rec.updated_at < cutoff
                and rec.importance <= max_importance
                and rec.access_count <= max_access_count
            )
            faded = (
                min_retention is not None
                and retention_of(rec, self.config) < min_retention
            )
            if stale or faded:
                if self.store.delete(rec.id):
                    deleted += 1
        if deleted:
            self._audit("PRUNE", user_id, detail={"deleted": deleted})
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
        records = self.store.list(
            user_id=user_id, limit=1_000_000, current_only=False
        )
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
        current = self.store.count(user_id=user_id)
        return {
            "total": current,
            "episodic": self.store.count(user_id=user_id,
                                         memory_type=MemoryType.EPISODIC),
            "semantic": self.store.count(user_id=user_id,
                                         memory_type=MemoryType.SEMANTIC),
            "archived": self.store.count(user_id=user_id, current_only=False)
            - current,
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

    def _audit(
        self,
        action: str,
        user_id: str | None,
        memory_id: str | None = None,
        reasoning: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.store.log_audit(
                AuditEntry(
                    action=action, user_id=user_id, memory_id=memory_id,
                    reasoning=reasoning, detail=detail or {},
                )
            )
        except Exception as exc:  # noqa: BLE001 - auditing must never break writes
            logger.error("audit logging failed: %s", exc)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.config.chars_per_token) + 1
