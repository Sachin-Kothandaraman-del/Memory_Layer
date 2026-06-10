"""Semantic fact extraction: distill durable facts from conversation events."""

from __future__ import annotations

import logging

from .llm import LLM
from .models import ExtractedFact, FactCategory

logger = logging.getLogger("memlayer")

_VALID_CATEGORIES = {c.value for c in FactCategory}

EXTRACTION_SYSTEM = """\
You are the memory-formation module of an AI agent. You read a conversation
excerpt and extract durable facts worth remembering across future sessions.

Extract ONLY information that will still be useful later:
- identity: who the user is (name, role, location, languages, tools they use)
- preference: likes, dislikes, communication/style preferences
- goal: what the user is trying to achieve (projects, deadlines, targets)
- relationship: people, teams, or organizations connected to the user
- constraint: hard rules the agent must respect ("never...", "always...")
- event: notable dated occurrences worth recalling
- knowledge: domain facts the agent was taught and should retain
- other: anything else genuinely durable

Do NOT extract:
- transient chit-chat, greetings, or one-off requests
- information the assistant generated itself (unless the user confirmed it)
- restatements of things that are obvious from the current request

Each fact must be a self-contained third-person sentence ("User's name is
Priya", not "my name is Priya"). Resolve pronouns. Merge near-duplicates.

Respond with JSON only:
{"facts": [{"content": str, "category": str, "importance": float 0.0-1.0}]}

importance: 0.9+ for identity/constraints, ~0.7 for goals/preferences,
~0.5 for events/knowledge. Return {"facts": []} if nothing qualifies.
"""


class FactExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, text: str) -> list[ExtractedFact]:
        """Extract semantic facts from a conversation excerpt or event text."""
        try:
            data = self.llm.generate_json(
                f"Conversation excerpt:\n---\n{text}\n---", system=EXTRACTION_SYSTEM
            )
        except Exception as exc:  # noqa: BLE001 - extraction must never crash writes
            logger.error("fact extraction failed: %s", exc)
            return []

        facts: list[ExtractedFact] = []
        for item in (data or {}).get("facts", []):
            content = (item.get("content") or "").strip()
            if not content:
                continue
            category = item.get("category", FactCategory.OTHER.value)
            if category not in _VALID_CATEGORIES:
                category = FactCategory.OTHER.value
            try:
                importance = float(item.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            facts.append(
                ExtractedFact(
                    content=content,
                    category=category,
                    importance=max(0.0, min(1.0, importance)),
                )
            )
        return facts
