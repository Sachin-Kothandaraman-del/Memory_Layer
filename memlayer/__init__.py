"""memlayer — persistent episodic + semantic memory middleware for LLM agents,
powered by the Gemini API."""

from .config import MemoryConfig, load_dotenv_file
from .core import MemoryLayer, MissingAPIKeyError
from .embeddings import GeminiEmbedder
from .llm import GeminiLLM
from .middleware import MemoryMiddleware, with_memory
from .models import (
    ExtractedFact,
    FactCategory,
    MemoryRecord,
    MemoryType,
    ScoredMemory,
)
from .storage import MemoryStore, SQLiteMemoryStore

__version__ = "0.1.0"

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
    "GeminiEmbedder",
    "GeminiLLM",
    "MemoryStore",
    "SQLiteMemoryStore",
]
