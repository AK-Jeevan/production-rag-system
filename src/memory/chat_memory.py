import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMemory:

    VALID_ROLES = {"user", "assistant", "system"}

    def __init__(self, max_history: int = 20):
        self.history     = []
        self.max_history = max_history

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"❌ Invalid role '{role}'. Must be one of: {self.VALID_ROLES}"
            )
        if not content or not isinstance(content, str):
            raise ValueError("❌ Content must be a non-empty string.")

        self.history.append({"role": role, "content": content})

        # Trim oldest messages if over limit (keep system message if present)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            logger.info(f"🔁 History trimmed to last {self.max_history} messages.")

        logger.info(f"💬 Added [{role}] message ({len(content)} chars).")

    def add_user_message(self, content: str) -> None:
        """Shortcut to add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Shortcut to add an assistant message."""
        self.add_message("assistant", content)

    def get_history(self) -> list:
        """Return full chat history."""
        return self.history

    def get_history_as_text(self) -> str:
        """Format history as a readable string for prompt injection."""
        if not self.history:
            return "No conversation history."

        lines = []
        for msg in self.history:
            role    = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def get_last_n(self, n: int) -> list:
        """Return the last n messages."""
        return self.history[-n:]

    def clear(self) -> None:
        """Clear all chat history."""
        self.history = []
        logger.info("🗑️  Chat history cleared.")

    def message_count(self) -> int:
        """Return total number of messages."""
        return len(self.history)

    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.history) == 0


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    memory = ChatMemory(max_history=10)

    # Add messages
    memory.add_user_message("What is FastAPI?")
    memory.add_assistant_message("FastAPI is a modern Python web framework for building APIs.")
    memory.add_user_message("How does it compare to Flask?")
    memory.add_assistant_message("FastAPI is faster and supports async out of the box.")

    print(f"\n--- Chat History ({memory.message_count()} messages) ---")
    for msg in memory.get_history():
        print(f"  [{msg['role']}]: {msg['content']}")

    print(f"\n--- History as Text ---")
    print(memory.get_history_as_text())

    print(f"\n--- Last 2 Messages ---")
    for msg in memory.get_last_n(2):
        print(f"  [{msg['role']}]: {msg['content']}")

    print(f"\n--- Stats ---")
    print(f"  Total messages : {memory.message_count()}")
    print(f"  Is empty       : {memory.is_empty()}")

    # Clear
    memory.clear()
    print(f"\n--- After Clear ---")
    print(f"  Total messages : {memory.message_count()}")
    print(f"  Is empty       : {memory.is_empty()}")