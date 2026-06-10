"""Memory consolidation: reconcile new facts against what is already stored.

For each candidate fact we look up the most similar existing semantic
memories. If nothing is close, the fact is added directly (no LLM call).
Otherwise the LLM decides between ADD / UPDATE / DELETE / NONE so the store
converges to a compact, current set of facts instead of accumulating
duplicates and contradictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .llm import LLM
from .models import MemoryRecord

logger = logging.getLogger("memlayer")

CONSOLIDATION_SYSTEM = """\
You maintain the long-term memory store of an AI agent. Given a NEW FACT and
the most similar EXISTING MEMORIES, decide how to integrate the new fact.

Operations:
- {"op": "ADD"}
    the new fact is genuinely new information; store it as-is.
- {"op": "UPDATE", "id": "<existing id>", "content": "<merged fact>"}
    the new fact refines, extends, or supersedes an existing memory;
    rewrite that memory so it reflects the latest truth.
- {"op": "DELETE", "id": "<existing id>"}
    an existing memory is now wrong or obsolete because of the new fact.
- {"op": "NONE"}
    the new fact is already fully covered; do nothing.

Rules:
- Prefer UPDATE over ADD+DELETE when one memory clearly supersedes another.
- Newer information wins over older information when they conflict.
- Multiple operations are allowed (e.g. UPDATE one memory and DELETE another).
- If you UPDATE or DELETE, you usually should NOT also ADD the same content.

Respond with JSON only: {"operations": [ ... ]}
"""


@dataclass
class ConsolidationResult:
    added: list[MemoryRecord]
    updated: list[MemoryRecord]
    deleted: list[str]
    skipped: bool = False  # NONE — fact already covered


class Consolidator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def decide(
        self, new_fact: str, similar: list[tuple[MemoryRecord, float]]
    ) -> list[dict]:
        """Ask the LLM for integration operations. Falls back to ADD on error."""
        existing = [
            {"id": rec.id, "content": rec.content, "similarity": round(sim, 3)}
            for rec, sim in similar
        ]
        prompt = (
            f"NEW FACT:\n{new_fact}\n\n"
            f"EXISTING MEMORIES:\n{json.dumps(existing, ensure_ascii=False, indent=2)}"
        )
        try:
            data = self.llm.generate_json(prompt, system=CONSOLIDATION_SYSTEM)
            ops = (data or {}).get("operations", [])
            return ops if isinstance(ops, list) else [{"op": "ADD"}]
        except Exception as exc:  # noqa: BLE001 - consolidation must not lose facts
            logger.error("consolidation failed, defaulting to ADD: %s", exc)
            return [{"op": "ADD"}]
