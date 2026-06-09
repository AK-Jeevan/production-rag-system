import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGISTRY_PATH = "config/prompt_registry.json"


class PromptRegistry:
    # ── Built-in versioned templates ─────────────────────────────────────────
    DEFAULT_TEMPLATES = {
        "rag_v1": {
            "version": "v1",
            "name": "rag_v1",
            "description": "Basic RAG prompt — answer only from context.",
            "template": """You are a helpful AI assistant.
Answer ONLY using the provided context below.
If the answer cannot be found in the context, say:
"I could not find the answer in the knowledge base."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:""",
            "created_at": "2024-01-01T00:00:00",
            "active": False,
        },
        "rag_v2": {
            "version": "v2",
            "name": "rag_v2",
            "description": "Domain-specific RAG prompt with MLOps expertise.",
            "template": """You are a helpful AI assistant with expertise in \
MLOps, LangChain, FastAPI, Docker, MLflow, DVC, and AWS.

Answer ONLY using the provided context below.
If the answer cannot be found in the context, say:
"I could not find the answer in the knowledge base."

Do NOT make up information. Be concise and accurate.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:""",
            "created_at": "2024-02-01T00:00:00",
            "active": False,
        },
        "rag_v3": {
            "version": "v3",
            "name": "rag_v3",
            "description": "RAG prompt with structured response and source citation.",
            "template": """You are a helpful AI assistant with expertise in \
MLOps, LangChain, FastAPI, Docker, MLflow, DVC, and AWS.

Answer ONLY using the provided context below.
If the answer cannot be found in the context, say:
"I could not find the answer in the knowledge base."

Structure your response as:
- Direct answer first
- Supporting details from context
- Cite sources where possible

Do NOT make up information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:""",
            "created_at": "2024-03-01T00:00:00",
            "active": True,  # ← default active version
        },
        "summary_v1": {
            "version": "v1",
            "name": "summary_v1",
            "description": "Summarization prompt.",
            "template": """You are a helpful AI assistant.
Summarize the following context in a clear and concise way.

CONTEXT:
{context}

SUMMARY:""",
            "created_at": "2024-01-01T00:00:00",
            "active": True,
        },
        "followup_v1": {
            "version": "v1",
            "name": "followup_v1",
            "description": "Follow-up prompt with conversation history.",
            "template": """You are a helpful AI assistant.
Given the conversation history and context, answer the follow-up question.

CONVERSATION HISTORY:
{history}

CONTEXT:
{context}

FOLLOW-UP QUESTION:
{query}

ANSWER:""",
            "created_at": "2024-01-01T00:00:00",
            "active": True,
        },
    }

    def __init__(self):
        self.registry = {}
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load registry from disk or initialize with defaults."""
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
            logger.info(f"📂 Prompt registry loaded: {len(self.registry)} templates.")
        else:
            self.registry = self.DEFAULT_TEMPLATES.copy()
            self._save()
            logger.info(
                f"✅ Prompt registry initialized with {len(self.registry)} default templates."
            )

    def _save(self) -> None:
        """Persist registry to disk."""
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=4)
        logger.info(f"💾 Prompt registry saved to: {REGISTRY_PATH}")

    def get(self, name: str) -> dict:
        """Get a prompt template by name."""
        if name not in self.registry:
            raise KeyError(f"❌ Prompt '{name}' not found in registry.")
        return self.registry[name]

    def get_template(self, name: str) -> str:
        """Get just the template string by name."""
        return self.get(name)["template"]

    def get_active(self, prefix: str = "rag") -> dict:
        """Get the active template for a given prefix (e.g. 'rag', 'summary')."""
        active = [
            v
            for k, v in self.registry.items()
            if k.startswith(prefix) and v.get("active", False)
        ]
        if not active:
            raise ValueError(f"❌ No active template found for prefix: '{prefix}'")

        # Return the most recently created active template
        return sorted(active, key=lambda x: x["created_at"], reverse=True)[0]

    def get_active_template(self, prefix: str = "rag") -> str:
        """Get just the active template string for a given prefix."""
        return self.get_active(prefix)["template"]

    def register(
        self,
        name: str,
        template: str,
        description: str = "",
        set_active: bool = False,
    ) -> None:
        """Register a new prompt template."""
        if not name or not template:
            raise ValueError("❌ Name and template must be non-empty strings.")

        if name in self.registry:
            logger.warning(f"⚠️  Overwriting existing prompt: '{name}'")

        self.registry[name] = {
            "version": name.split("_")[-1] if "_" in name else "v1",
            "name": name,
            "description": description,
            "template": template,
            "created_at": datetime.utcnow().isoformat(),
            "active": set_active,
        }

        # If set_active, deactivate other prompts with same prefix
        if set_active:
            prefix = name.rsplit("_", 1)[0]
            self._deactivate_others(name, prefix)

        self._save()
        logger.info(f"✅ Prompt '{name}' registered. Active: {set_active}")

    def set_active(self, name: str) -> None:
        """Set a prompt as active and deactivate others with same prefix."""
        if name not in self.registry:
            raise KeyError(f"❌ Prompt '{name}' not found in registry.")

        prefix = name.rsplit("_", 1)[0]
        self._deactivate_others(name, prefix)

        self.registry[name]["active"] = True
        self._save()
        logger.info(f"✅ Prompt '{name}' set as active.")

    def _deactivate_others(self, active_name: str, prefix: str) -> None:
        """Deactivate all prompts with same prefix except active_name."""
        for key in self.registry:
            if key.startswith(prefix) and key != active_name:
                self.registry[key]["active"] = False

    def list_all(self) -> None:
        """Print all registered prompts."""
        print(f"\n{'=' * 70}")
        print(f"  📋 Prompt Registry ({len(self.registry)} templates)")
        print(f"{'=' * 70}")
        for name, meta in self.registry.items():
            active = "✅ ACTIVE" if meta.get("active") else "  inactive"
            print(f"  {active} | {name:20} | {meta['description']}")
        print(f"{'=' * 70}\n")

    def delete(self, name: str) -> None:
        """Delete a prompt from the registry."""
        if name not in self.registry:
            raise KeyError(f"❌ Prompt '{name}' not found.")
        del self.registry[name]
        self._save()
        logger.info(f"🗑️  Prompt '{name}' deleted.")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    registry = PromptRegistry()

    # List all
    registry.list_all()

    # Get active RAG template
    print("--- Active RAG Template ---")
    active = registry.get_active("rag")
    print(f"Name       : {active['name']}")
    print(f"Version    : {active['version']}")
    print(f"Description: {active['description']}")
    print(f"Template   :\n{active['template']}")

    # Register a new version
    print("\n--- Registering rag_v4 ---")
    registry.register(
        name="rag_v4",
        description="RAG v4 — step-by-step reasoning.",
        template="""You are a helpful AI assistant with expertise in MLOps.

Answer ONLY using the provided context below.
If the answer cannot be found, say:
"I could not find the answer in the knowledge base."

Think step by step before answering.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:""",
        set_active=True,
    )

    registry.list_all()

    # Switch back to v3
    print("--- Switching back to rag_v3 ---")
    registry.set_active("rag_v3")
    registry.list_all()
