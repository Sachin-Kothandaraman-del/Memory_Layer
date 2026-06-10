"""Minimal end-to-end demo of the memory layer (requires GEMINI_API_KEY).

Run:  python examples/quickstart.py
"""

from memlayer import MemoryLayer


def main() -> None:
    with MemoryLayer.from_env(db_path="quickstart.db") as mem:
        # 1. Store some events — semantic facts are extracted automatically.
        print("Writing memories...")
        for line in [
            "user: Hi! I'm Sachin, a backend developer in Bangalore.",
            "user: I'm building a SaaS analytics product called PulseBoard.",
            "user: Please always answer with code examples in Python, never Java.",
            "user: My co-founder Ananya handles the frontend in React.",
        ]:
            result = mem.add(line, user_id="sachin")
            for fact in result["facts"]:
                print(f"  extracted: {fact['content']}")

        # 2. Later (any session, any process) — retrieve relevant context.
        query = "help me design an API endpoint for my product"
        print(f"\nQuery: {query}\n")
        for hit in mem.search(query, user_id="sachin", limit=5):
            print(f"  {hit.score:.3f}  [{hit.record.memory_type.value:8}] "
                  f"{hit.record.content}")

        # 3. Or get a ready-to-inject prompt block.
        print("\n--- context block ---")
        print(mem.get_context(query, user_id="sachin", token_budget=500))

        print("\nStats:", mem.stats(user_id="sachin"))


if __name__ == "__main__":
    main()
