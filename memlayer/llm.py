"""LLM provider for extraction / consolidation / summarization. Default: Gemini."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Protocol

logger = logging.getLogger("memlayer")


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...
    def generate_json(self, prompt: str, system: str | None = None) -> Any: ...


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_loosely(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = _JSON_FENCE.search(text)
    if fence:
        return json.loads(fence.group(1))
    # last resort: first {...} or [...] span
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("no JSON found in model output", text, 0)


class GeminiLLM:
    """Thin Gemini text client with JSON mode and retry."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 5,
        temperature: float = 0.0,
        client=None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        if client is not None:
            self._client = client
        else:
            from google import genai  # lazy import

            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self._call(prompt, system, json_mode=False)

    def generate_json(self, prompt: str, system: str | None = None) -> Any:
        return parse_json_loosely(self._call(prompt, system, json_mode=True))

    def _call(self, prompt: str, system: str | None, json_mode: bool) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                sleep_for = delay + random.uniform(0, delay / 2)
                logger.warning(
                    "generate_content failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, sleep_for,
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")  # pragma: no cover
