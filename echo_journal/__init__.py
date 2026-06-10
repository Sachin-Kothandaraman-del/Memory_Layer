"""Echo — a journal you talk to, that lets you talk to your past self.

A consumer app built on memlayer's primitives:
- Memory Rescue: the forgetting curve surfaces memories about to fade;
  keeping one reinforces it (strength grows), exactly like human rehearsal
- Past Self: time-travel queries answer with what you believed *then*
- Insights: evidence-cited weekly reflections, not horoscope fluff
- Glass box: every memory's story (versions, sources, reasoning) is one tap away
- Your life is a file you own: local SQLite, PII guards, "off the record"
"""

from .server import EchoServer, create_server, main

__all__ = ["EchoServer", "create_server", "main"]
