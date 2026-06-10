"""memlayer — persistent episodic + semantic memory middleware for LLM agents,
powered by the Gemini API."""

from .config import MemoryConfig, load_dotenv_file
from .core import MemoryLayer, MissingAPIKeyError
from .embeddings import GeminiEmbedder
from .llm import GeminiLLM
from .middleware import MemoryMiddleware, with_memory
from .models import (
    AuditEntry,
    ExtractedFact,
    FactCategory,
    MemoryRecord,
    MemoryType,
    ScoredMemory,
)
from .privacy import matches_never_remember, redact_pii
from .reflection import Insight, Reflector
from .storage import MemoryStore, SQLiteMemoryStore

__version__ = "0.2.0"

__all__ = [
    "MemoryLayer",
    "MemoryConfig",
    "MissingAPIKeyError",
    "load_dotenv_file",
    "MemoryMiddleware",
    "with_memory",
    "MemoryRecord",
    "MemoryType",
    "FactCategory",
    "ScoredMemory",
    "ExtractedFact",
    "AuditEntry",
    "Insight",
    "Reflector",
    "redact_pii",
    "matches_never_remember",
    "GeminiEmbedder",
    "GeminiLLM",
    "MemoryStore",
    "SQLiteMemoryStore",
]
