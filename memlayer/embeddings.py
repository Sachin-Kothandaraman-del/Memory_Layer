"""Embedding providers. Default: Gemini embeddings via google-genai."""

from __future__ import annotations

import logging
import random
import time
from collections import OrderedDict
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger("memlayer")


class Embedder(Protocol):
    """Anything that can embed documents and queries into unit vectors."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def normalize(vec: Sequence[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


class GeminiEmbedder:
    """Gemini embedding client with batching, retry, and an LRU cache.

    Vectors are L2-normalized so cosine similarity is a plain dot product
    (required: truncated Gemini embeddings are not pre-normalized).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        batch_size: int = 100,
        max_retries: int = 5,
        cache_size: int = 4096,
        client=None,
    ):
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._cache_size = cache_size
        if client is not None:
            self._client = client
        else:
            from google import genai  # lazy: tests/custom embedders skip the dep

            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # -- public API ---------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    # -- internals ----------------------------------------------------------

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache_get((task_type, text))
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)

        for start in range(0, len(misses), self.batch_size):
            chunk = misses[start : start + self.batch_size]
            vectors = self._call_api([texts[i] for i in chunk], task_type)
            for i, vec in zip(chunk, vectors):
                results[i] = vec
                self._cache_put((task_type, texts[i]), vec)

        return results  # type: ignore[return-value]

    def _call_api(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dim,
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                return [normalize(e.values) for e in resp.embeddings]
            except Exception as exc:  # noqa: BLE001 - SDK raises varied types
                if attempt == self.max_retries - 1:
                    raise
                sleep_for = delay + random.uniform(0, delay / 2)
                logger.warning(
                    "embed_content failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, sleep_for,
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _cache_get(self, key: tuple[str, str]) -> list[float] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: tuple[str, str], value: list[float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
