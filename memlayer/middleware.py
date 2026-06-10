"""Drop-in middleware: add persistent memory to any chat function.

Works with any agent that exchanges OpenAI-style messages
(``[{"role": ..., "content": ...}, ...]``):

    middleware = MemoryMiddleware(memory, user_id="u1")
    chat = middleware.wrap(my_chat_fn)          # my_chat_fn(messages) -> str
    reply = chat([{"role": "user", "content": "hi"}])

or hook the two phases manually:

    messages = middleware.before(messages)      # inject recalled context
    reply = call_llm(messages)
    middleware.after(messages, reply)           # record the exchange
"""

from __future__ import annotations

import functools
import uuid
from typing import Callable, Sequence

from .core import MemoryLayer

Message = dict[str, str]
ChatFn = Callable[[list[Message]], str]


class MemoryMiddleware:
    def __init__(
        self,
        memory: MemoryLayer,
        *,
        user_id: str = "default",
        agent_id: str | None = None,
        session_id: str | None = None,
        token_budget: int | None = None,
        background_writes: bool = True,
        record_assistant: bool = True,
    ):
        self.memory = memory
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.token_budget = token_budget
        self.background_writes = background_writes
        self.record_assistant = record_assistant

    # -- phase 1: before inference -------------------------------------------

    def before(self, messages: Sequence[Message]) -> list[Message]:
        """Retrieve relevant memories for the latest user message and inject
        them as (part of) the system message. Returns a new message list."""
        messages = [dict(m) for m in messages]
        query = self._latest_user_content(messages)
        if not query:
            return messages

        context = self.memory.get_context(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            token_budget=self.token_budget,
        )
        if not context:
            return messages

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{context}"
        else:
            messages.insert(0, {"role": "system", "content": context})
        return messages

    # -- phase 2: after inference ---------------------------------------------

    def after(self, messages: Sequence[Message], response: str) -> None:
        """Record the exchange (latest user message + assistant reply)."""
        user_msg = self._latest_user_content(messages)
        if not user_msg and not response:
            return
        turn = [{"role": "user", "content": user_msg}]
        if self.record_assistant and response:
            turn.append({"role": "assistant", "content": response})
        self.memory.add_messages(
            turn,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            wait=not self.background_writes,
        )

    # -- one-call integration ---------------------------------------------------

    def wrap(self, chat_fn: ChatFn) -> ChatFn:
        """Wrap ``chat_fn(messages) -> str`` with memory recall + recording."""

        @functools.wraps(chat_fn)
        def wrapped(messages: list[Message]) -> str:
            augmented = self.before(messages)
            response = chat_fn(augmented)
            self.after(messages, response)
            return response

        return wrapped

    def flush(self) -> None:
        self.memory.flush()

    @staticmethod
    def _latest_user_content(messages: Sequence[Message]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


def with_memory(
    memory: MemoryLayer,
    *,
    user_id: str = "default",
    **middleware_kwargs,
) -> Callable[[ChatFn], ChatFn]:
    """Decorator form::

        @with_memory(mem, user_id="u1")
        def chat(messages): ...
    """
    middleware = MemoryMiddleware(memory, user_id=user_id, **middleware_kwargs)

    def decorator(chat_fn: ChatFn) -> ChatFn:
        return middleware.wrap(chat_fn)

    return decorator
