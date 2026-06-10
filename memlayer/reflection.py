"""Reflection ("sleep" consolidation): periodically review recent experience
and distill higher-order insights the per-message extractor can't see.

Where extraction pulls facts out of a single message, reflection looks
across many episodes at once and asks "what patterns emerge?" — recurring
preferences, behavioral tendencies, connections between events. Each insight
cites the episode ids it was derived from (evidence), keeping the glass-box
property: every reflected belief is traceable to raw experience.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .llm import LLM
from .models import MemoryRecord

logger = logging.getLogger("memlayer")

REFLECTION_SYSTEM = """\
You are the reflection module of an AI agent's long-term memory — the
equivalent of consolidating experience during sleep. You receive the agent's
RECENT EPISODES (raw conversation events, each with an id) and its KNOWN
FACTS (what has already been distilled).

Identify higher-order insights that are NOT already captured in the known
facts:
- recurring preferences, habits, or behavioral patterns across episodes
- connections between separate events ("the deadline stress and the late
  replies are about the same project")
- tendencies worth knowing ("user asks for code first, explanation second")
- contradictions between what the facts say and what the episodes show

Each insight must cite the episode ids it is derived from as evidence.
Set a high bar: an insight must be supported by MULTIPLE episodes or a
clear pattern, be genuinely useful for serving this user better, and not
restate an existing known fact. It is normal to find nothing.

Respond with JSON only:
{"insights": [{"insight": str, "evidence": [episode ids],
               "confidence": float 0.0-1.0, "reasoning": str}]}
"""


@dataclass
class Insight:
    content: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.6
    reasoning: str = ""


class Reflector:
    def __init__(self, llm: LLM):
        self.llm = llm

    def reflect(
        self,
        episodes: list[MemoryRecord],
        known_facts: list[MemoryRecord],
    ) -> list[Insight]:
        if not episodes:
            return []
        episode_payload = [
            {"id": e.id, "when": e.created_at, "event": e.content}
            for e in sorted(episodes, key=lambda r: r.created_at)
        ]
        facts_payload = [f.content for f in known_facts]
        prompt = (
            "RECENT EPISODES:\n"
            + json.dumps(episode_payload, ensure_ascii=False, indent=1)
            + "\n\nKNOWN FACTS:\n"
            + json.dumps(facts_payload, ensure_ascii=False, indent=1)
        )
        try:
            data = self.llm.generate_json(prompt, system=REFLECTION_SYSTEM)
        except Exception as exc:  # noqa: BLE001 - reflection is best-effort
            logger.error("reflection failed: %s", exc)
            return []

        valid_ids = {e.id for e in episodes}
        insights: list[Insight] = []
        for item in (data or {}).get("insights", []):
            content = (item.get("insight") or "").strip()
            if not content:
                continue
            evidence = [
                e for e in (item.get("evidence") or []) if e in valid_ids
            ]
            try:
                confidence = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            insights.append(
                Insight(
                    content=content,
                    evidence=evidence,
                    confidence=max(0.0, min(1.0, confidence)),
                    reasoning=(item.get("reasoning") or "").strip(),
                )
            )
        return insights
