"""Hybrid retrieval: vector + keyword search, fused and re-ranked, with an
Ebbinghaus-style forgetting curve.

Pipeline:
1. vector search (Gemini embeddings, cosine) and FTS5 keyword search
2. reciprocal-rank fusion of the two candidate lists
3. composite scoring: similarity * w1 + retention * w2 + importance * w3
4. forgetting: memories whose retention fell below the floor have "faded"
   and are excluded (not deleted — they can be recovered or pruned later)
5. MMR (maximal marginal relevance) to remove near-duplicate results
6. reinforcement: returned memories are "touched", which bumps access stats
   and multiplies strength — recalled memories decay slower next time

Retention model: ``retention = 0.5 ** (hours_since_last_access / half_life)``
where ``half_life = base_half_life * strength * (0.5 + 1.5 * importance)``.
Important, frequently-recalled memories become near-permanent; trivia that
is never recalled evaporates in weeks.
"""

from __future__ import annotations

import time

import numpy as np

from .config import MemoryConfig
from .embeddings import Embedder
from .models import MemoryRecord, MemoryType, ScoredMemory
from .storage.base import MemoryStore


def retention_of(
    record: MemoryRecord, config: MemoryConfig, now: float | None = None
) -> float:
    """Current retention (0..1) of a memory under the forgetting curve."""
    now = now or time.time()
    last = max(record.last_accessed_at, record.updated_at)
    age_hours = max(0.0, (now - last) / 3600.0)
    half_life = config.recency_half_life_hours
    if config.enable_forgetting:
        half_life *= record.strength * (0.5 + 1.5 * record.importance)
    return 0.5 ** (age_hours / max(half_life, 1e-6))


class Retriever:
    def __init__(self, store: MemoryStore, embedder: Embedder, config: MemoryConfig):
        self.store = store
        self.embedder = embedder
        self.config = config

    def search(
        self,
        query: str,
        limit: int = 8,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        reinforce: bool = True,
        include_faded: bool = False,
        as_of: float | None = None,
    ) -> list[ScoredMemory]:
        cfg = self.config
        pool = max(cfg.candidate_pool, limit * 3)
        if as_of is not None:
            # historical queries are read-only and bypass the forgetting filter
            reinforce = False
            include_faded = True
        filters = dict(
            user_id=user_id, agent_id=agent_id,
            session_id=session_id, memory_type=memory_type, as_of=as_of,
        )

        query_vec = self.embedder.embed_query(query)
        vector_hits = self.store.vector_search(query_vec, limit=pool, **filters)
        keyword_hits = self.store.keyword_search(query, limit=pool, **filters)

        candidates = self._fuse(vector_hits, keyword_hits)
        if not candidates:
            return []

        now = time.time()
        scored = [self._score(rec, sim, now) for rec, sim in candidates.values()]
        if cfg.enable_forgetting and not include_faded:
            scored = [s for s in scored if s.recency >= cfg.retention_floor]
        scored.sort(key=lambda s: s.score, reverse=True)
        results = self._mmr(scored, limit)

        if reinforce and results:
            self.store.touch(
                [s.record.id for s in results],
                strength_factor=cfg.strength_reinforce_factor,
                strength_max=cfg.strength_max,
            )
        return results

    # -- pipeline stages ------------------------------------------------------

    def _fuse(
        self,
        vector_hits: list[tuple[MemoryRecord, float]],
        keyword_hits: list[MemoryRecord],
    ) -> dict[str, tuple[MemoryRecord, float]]:
        """Reciprocal-rank fusion. Keeps the best-known cosine sim per record."""
        k = self.config.rrf_k
        rrf: dict[str, float] = {}
        records: dict[str, tuple[MemoryRecord, float]] = {}

        for rank, (rec, sim) in enumerate(vector_hits):
            rrf[rec.id] = rrf.get(rec.id, 0.0) + 1.0 / (k + rank + 1)
            records[rec.id] = (rec, sim)
        for rank, rec in enumerate(keyword_hits):
            rrf[rec.id] = rrf.get(rec.id, 0.0) + 1.0 / (k + rank + 1)
            if rec.id not in records:
                records[rec.id] = (rec, 0.0)  # keyword-only hit: no cosine known

        # order candidates by fused rank so _score sees the strongest first
        ordered = sorted(records, key=lambda i: rrf[i], reverse=True)
        return {i: records[i] for i in ordered}

    def _score(
        self, record: MemoryRecord, similarity: float, now: float
    ) -> ScoredMemory:
        cfg = self.config
        retention = retention_of(record, cfg, now)
        sim = max(0.0, min(1.0, similarity))
        score = (
            cfg.weight_similarity * sim
            + cfg.weight_recency * retention
            + cfg.weight_importance * record.importance
        )
        return ScoredMemory(
            record=record,
            similarity=sim,
            recency=retention,
            importance=record.importance,
            score=score,
        )

    def _mmr(self, scored: list[ScoredMemory], limit: int) -> list[ScoredMemory]:
        """Maximal marginal relevance: relevance minus redundancy."""
        lam = self.config.mmr_lambda
        if lam >= 1.0 or len(scored) <= limit:
            return scored[:limit]

        selected: list[ScoredMemory] = []
        remaining = list(scored)
        while remaining and len(selected) < limit:
            best, best_val = None, -np.inf
            for cand in remaining:
                redundancy = max(
                    (self._sim(cand.record, s.record) for s in selected),
                    default=0.0,
                )
                val = lam * cand.score - (1.0 - lam) * redundancy
                if val > best_val:
                    best, best_val = cand, val
            selected.append(best)  # type: ignore[arg-type]
            remaining.remove(best)  # type: ignore[arg-type]
        return selected

    @staticmethod
    def _sim(a: MemoryRecord, b: MemoryRecord) -> float:
        if a.embedding is None or b.embedding is None:
            return 0.0
        va = np.asarray(a.embedding, dtype=np.float32)
        vb = np.asarray(b.embedding, dtype=np.float32)
        return float(va @ vb)  # stored vectors are normalized
