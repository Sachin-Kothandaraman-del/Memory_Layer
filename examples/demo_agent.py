"""Interactive CLI chat agent with persistent memory (requires GEMINI_API_KEY).

The agent itself is three lines of Gemini code; everything memory-related
comes from wrapping it with MemoryMiddleware. Quit and re-run the script —
it remembers you across sessions.

Run:  python examples/demo_agent.py [user_id]
"""

import sys

from google import genai

from memlayer import MemoryLayer, MemoryMiddleware, load_dotenv_file

load_dotenv_file()  # built-in .env loader (no python-dotenv needed)

SYSTEM = "You are a helpful, concise personal assistant."


def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo-user"
    client = genai.Client()

    def chat_fn(messages: list[dict]) -> str:
        """Plain Gemini call — knows nothing about memory."""
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config={"system_instruction": system or SYSTEM},
        )
        return resp.text or ""

    memory = MemoryLayer.from_env(db_path="agent_memory.db")
    middleware = MemoryMiddleware(memory, user_id=user_id)
    chat = middleware.wrap(chat_fn)

    history: list[dict] = []
    print(f"Chatting as '{user_id}' — type 'quit' to exit. "
          f"({memory.stats(user_id=user_id)['total']} memories on file)")
    try:
        while True:
            user_input = input("\nyou> ").strip()
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            history.append({"role": "user", "content": user_input})
            reply = chat(list(history))
            history.append({"role": "assistant", "content": reply})
            print(f"\nagent> {reply}")
    finally:
        memory.close()  # flushes pending background writes
        print("\nMemories saved. See you next session!")


if __name__ == "__main__":
    main()
