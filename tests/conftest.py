"""Test doubles: a deterministic embedder and a scriptable LLM.

FakeEmbedder hashes word unigrams into a 64-dim bag-of-words vector, so
texts that share vocabulary really are cosine-similar — retrieval tests
exercise the actual ranking math without network calls.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

import numpy as np
import pytest

from memlayer import MemoryConfig, MemoryLayer
from memlayer.storage.sqlite_store import SQLiteMemoryStore


class FakeEmbedder:
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Returns queued JSON responses; falls back to inert defaults."""

    def __init__(self):
        self.json_queue: list[Any] = []
        self.text_queue: list[str] = []
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append({"kind": "text", "prompt": prompt, "system": system})
        return self.text_queue.pop(0) if self.text_queue else "ok"

    def generate_json(self, prompt: str, system: str | None = None) -> Any:
        self.calls.append({"kind": "json", "prompt": prompt, "system": system})
        if self.json_queue:
            return self.json_queue.pop(0)
        if system and "memory-formation" in system:
            return {"facts": []}
        if system and "reflection module" in system:
            return {"insights": []}
        return {"operations": [{"op": "ADD"}]}


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store() -> SQLiteMemoryStore:
    s = SQLiteMemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def memory(fake_embedder, fake_llm) -> MemoryLayer:
    config = MemoryConfig(db_path=":memory:", embed_dim=FakeEmbedder.dim)
    mem = MemoryLayer(config=config, embedder=fake_embedder, llm=fake_llm)
    yield mem
    mem.close()
