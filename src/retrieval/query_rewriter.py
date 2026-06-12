import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configure Gemini ──────────────────────────────────────────────────────────
def _get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")
    return genai.Client(api_key=api_key)


class QueryRewriter:
    REWRITE_TEMPLATE = """You are an expert at reformulating search queries to \
improve document retrieval.

Given the original question and conversation history, rewrite the question to:
1. Be more specific and detailed
2. Include relevant keywords for better retrieval
3. Resolve any pronouns or references using conversation history
4. Be self-contained (understandable without history)

Return ONLY the rewritten query. No explanation, no preamble, no quotes.

CONVERSATION HISTORY:
{history}

ORIGINAL QUESTION:
{question}

REWRITTEN QUESTION:"""

    EXPAND_TEMPLATE = """You are an expert at expanding search queries.

Given the original question, generate {n} different versions of the question
that capture the same intent but use different keywords and phrasing.
This helps retrieve more relevant documents.

Return ONLY the {n} versions, one per line. No numbering, no explanation.

ORIGINAL QUESTION:
{question}

EXPANDED QUESTIONS:"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.client = _get_client()
        self.config = types.GenerateContentConfig(
            temperature=0.3,  # low temp for focused rewrites
            max_output_tokens=256,
        )
        logger.info(f"✅ QueryRewriter loaded: {self.model_name}")

    def rewrite(self, question: str, history: str = "") -> str:
        """Rewrite a query using conversation history for context."""
        if not question or not isinstance(question, str):
            raise ValueError("❌ Question must be a non-empty string.")

        history_text = history if history else "No prior conversation."

        prompt = self.REWRITE_TEMPLATE.format(history=history_text, question=question)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            rewritten = response.text.strip()
            logger.info("✅ Query rewritten.")
            logger.info(f"   Original : {question}")
            logger.info(f"   Rewritten: {rewritten}")
            return rewritten

        except Exception as e:
            logger.error(f"❌ Query rewrite failed: {e}. Using original query.")
            return question  # fallback to original on error

    def expand(self, question: str, n: int = 3) -> list:
        """Generate n alternative versions of the query for better retrieval."""
        if not question or not isinstance(question, str):
            raise ValueError("❌ Question must be a non-empty string.")

        prompt = self.EXPAND_TEMPLATE.format(question=question, n=n)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            raw = response.text.strip()
            expansions = [line.strip() for line in raw.splitlines() if line.strip()][:n]

            # Always include original
            all_queries = [question] + expansions

            logger.info(f"✅ Query expanded into {len(all_queries)} versions.")
            for i, q in enumerate(all_queries):
                logger.info(f"   [{i}] {q}")

            return all_queries

        except Exception as e:
            logger.error(f"❌ Query expansion failed: {e}. Using original query.")
            return [question]  # fallback to original on error

    def rewrite_and_expand(self, question: str, history: str = "", n: int = 3) -> list:
        """Rewrite query using history, then expand into n versions."""
        rewritten = self.rewrite(question, history)
        all_queries = self.expand(rewritten, n=n)
        return all_queries


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rewriter = QueryRewriter()

    # Basic rewrite
    print("\n--- Basic Rewrite ---")
    rewritten = rewriter.rewrite(
        question="How do I use it?",
        history="User: What is FastAPI?\nAssistant: FastAPI is a Python web framework.",
    )
    print(f"Rewritten: {rewritten}")

    # Expand
    print("\n--- Query Expansion ---")
    expansions = rewriter.expand(question="What is MLflow used for?", n=3)
    for i, q in enumerate(expansions):
        print(f"  [{i}] {q}")

    # Rewrite + Expand
    print("\n--- Rewrite + Expand ---")
    all_queries = rewriter.rewrite_and_expand(
        question="How does it track experiments?",
        history="User: What is MLflow?\nAssistant: MLflow is an ML lifecycle platform.",
        n=3,
    )
    for i, q in enumerate(all_queries):
        print(f"  [{i}] {q}")
