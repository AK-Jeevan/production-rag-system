import logging
from src.generation.prompt_registry import PromptRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

registry = PromptRegistry()


class PromptBuilder:
    # ── Builders ──────────────────────────────────────────────────────────────
    @staticmethod
    def _format_context(retrieved_docs: list) -> str:
        """Format retrieved docs into a numbered context string."""
        if not retrieved_docs:
            return "No context available."

        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            parts.append(f"[{i}] Source: {source}\n{content}")

        return "\n\n".join(parts)

    @classmethod
    def build_prompt(
        cls,
        query: str,
        retrieved_docs: list,
        prompt_name: str = None,
    ) -> str:
        """Build a standard RAG prompt using active or specified template."""
        if not query:
            raise ValueError("❌ Query must be a non-empty string.")
        if not retrieved_docs:
            logger.warning("⚠️  No retrieved docs provided — context will be empty.")

        context = cls._format_context(retrieved_docs)

        # Use specified template or fall back to active RAG template
        template = (
            registry.get_template(prompt_name)
            if prompt_name
            else registry.get_active_template("rag")
        )

        prompt = template.format(context=context, query=query)
        logger.info(
            f"✅ RAG prompt built using: "
            f"{'active RAG' if not prompt_name else prompt_name} "
            f"({len(prompt)} chars)."
        )
        return prompt

    @classmethod
    def build_summary_prompt(
        cls,
        retrieved_docs: list,
        prompt_name: str = None,
    ) -> str:
        """Build a summarization prompt from retrieved docs."""
        if not retrieved_docs:
            raise ValueError("❌ No docs provided for summarization.")

        context = cls._format_context(retrieved_docs)
        template = (
            registry.get_template(prompt_name)
            if prompt_name
            else registry.get_active_template("summary")
        )

        prompt = template.format(context=context)
        logger.info(f"✅ Summary prompt built ({len(prompt)} chars).")
        return prompt

    @classmethod
    def build_followup_prompt(
        cls,
        query: str,
        retrieved_docs: list,
        history: list = None,
        prompt_name: str = None,
    ) -> str:
        """Build a follow-up prompt with conversation history."""
        if not query:
            raise ValueError("❌ Query must be a non-empty string.")

        context = cls._format_context(retrieved_docs)
        history_str = "\n".join(history) if history else "No prior conversation."
        template = (
            registry.get_template(prompt_name)
            if prompt_name
            else registry.get_active_template("followup")
        )

        prompt = template.format(context=context, query=query, history=history_str)
        logger.info(f"✅ Follow-up prompt built ({len(prompt)} chars).")
        return prompt

    @staticmethod
    def get_prompt_stats(prompt: str) -> dict:
        """Return basic stats about the prompt."""
        return {
            "total_chars": len(prompt),
            "total_words": len(prompt.split()),
            "total_lines": len(prompt.splitlines()),
            "approx_tokens": len(prompt) // 4,
        }

    @staticmethod
    def get_active_prompt_name() -> str:
        """Return the name of the currently active RAG prompt."""
        active = registry.get_active("rag")
        return active["name"]

    @staticmethod
    def list_prompts() -> None:
        """List all available prompts in the registry."""
        registry.list_all()


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    sample_docs = [
        Document(
            page_content="FastAPI is a modern, fast web framework for building APIs with Python.",
            metadata={"source": "fastapi.md"},
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by LLMs.",
            metadata={"source": "langchain.md"},
        ),
        Document(
            page_content="MLflow is an open source platform for managing the ML lifecycle.",
            metadata={"source": "mlflow.md"},
        ),
    ]

    builder = PromptBuilder()

    # List all prompts
    print("--- All Prompts in Registry ---")
    builder.list_prompts()

    # Build with active template
    print("--- Active RAG Prompt ---")
    prompt = builder.build_prompt(query="What is FastAPI?", retrieved_docs=sample_docs)
    print(prompt)

    stats = builder.get_prompt_stats(prompt)
    print("\n--- Prompt Stats ---")
    for key, value in stats.items():
        print(f"  {key:15}: {value}")

    # Build with specific version
    print("\n--- Specific Version (rag_v1) ---")
    prompt_v1 = builder.build_prompt(
        query="What is FastAPI?", retrieved_docs=sample_docs, prompt_name="rag_v1"
    )
    print(prompt_v1)

    # Summary prompt
    print("\n--- Summary Prompt ---")
    summary_prompt = builder.build_summary_prompt(sample_docs)
    print(summary_prompt)

    # Follow-up prompt
    print("\n--- Follow-up Prompt ---")
    history = [
        "User: What is FastAPI?",
        "Assistant: FastAPI is a modern web framework for building APIs with Python.",
    ]
    followup_prompt = builder.build_followup_prompt(
        query="How does it compare to Flask?",
        retrieved_docs=sample_docs,
        history=history,
    )
    print(followup_prompt)

    # Active prompt name
    print("\n--- Active Prompt ---")
    print(f"Active RAG prompt: {builder.get_active_prompt_name()}")
