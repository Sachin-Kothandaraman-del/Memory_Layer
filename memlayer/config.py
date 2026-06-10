"""Configuration for the memory layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def load_dotenv_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Tiny built-in parser (no python-dotenv dependency). Existing environment
    variables always win; missing or unreadable files are silently ignored.
    """
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


@dataclass
class MemoryConfig:
    """All tunable knobs for a :class:`memlayer.MemoryLayer` instance."""

    # --- Gemini API ---
    api_key: str | None = None  # falls back to GEMINI_API_KEY / GOOGLE_API_KEY
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    embed_batch_size: int = 100
    max_retries: int = 5

    # --- Storage ---
    db_path: str = "memlayer.db"  # ":memory:" for ephemeral

    # --- Extraction ---
    extract_semantic: bool = True          # run LLM fact extraction on writes
    min_extraction_chars: int = 12         # skip extraction for trivial inputs

    # --- Consolidation ---
    consolidate: bool = True               # reconcile new facts against old ones
    consolidation_top_k: int = 5           # similar facts shown to the LLM
    consolidation_sim_threshold: float = 0.75  # below this, ADD without an LLM call

    # --- Retrieval scoring ---
    weight_similarity: float = 0.65
    weight_recency: float = 0.15
    weight_importance: float = 0.20
    recency_half_life_hours: float = 24.0 * 7  # one week
    rrf_k: int = 60                        # reciprocal-rank-fusion constant
    mmr_lambda: float = 0.7                # 1.0 = pure relevance, 0.0 = pure diversity
    candidate_pool: int = 40               # candidates fetched before re-ranking

    # --- Context assembly ---
    default_token_budget: int = 1200
    chars_per_token: float = 4.0           # cheap token estimate

    # --- Hygiene ---
    episodic_default_importance: float = 0.3
    semantic_default_importance: float = 0.6

    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        return (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    @classmethod
    def from_env(cls, **overrides) -> "MemoryConfig":
        """Build a config from environment variables (and ./.env), with
        overrides."""
        load_dotenv_file()
        cfg = cls(**overrides)
        if cfg.api_key is None:
            cfg.api_key = cfg.resolve_api_key()
        if "MEMLAYER_DB_PATH" in os.environ and "db_path" not in overrides:
            cfg.db_path = os.environ["MEMLAYER_DB_PATH"]
        if "MEMLAYER_LLM_MODEL" in os.environ and "llm_model" not in overrides:
            cfg.llm_model = os.environ["MEMLAYER_LLM_MODEL"]
        if "MEMLAYER_EMBED_MODEL" in os.environ and "embed_model" not in overrides:
            cfg.embed_model = os.environ["MEMLAYER_EMBED_MODEL"]
        return cfg
