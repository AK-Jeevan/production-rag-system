import os
import logging
from typing import Generator
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configure Gemini ──────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)


class GeminiGenerator:
    AVAILABLE_MODELS = {
        "flash": "gemini-2.0-flash",
        "flash25": "gemini-2.5-flash",
        "pro": "gemini-1.5-pro",
    }

    def __init__(self, model_key: str = "flash25", temperature: float = 0.7):
        self.model_key = model_key
        self.model_name = self.AVAILABLE_MODELS.get(model_key, model_key)
        self.temperature = temperature

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=1024,
            ),
        )

        logger.info(f"✅ Gemini model loaded: {self.model_name}")

    def generate_answer(self, prompt: str) -> str:
        """Generate a full answer for a given prompt (non-streaming)."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("❌ Prompt must be a non-empty string.")

        logger.info(f"🔄 Generating answer for prompt ({len(prompt)} chars)...")

        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            logger.info(f"✅ Answer generated ({len(answer)} chars).")
            return answer

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise

    def generate_answer_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate a streaming answer token by token for a given prompt."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("❌ Prompt must be a non-empty string.")

        logger.info(f"🔄 Streaming answer for prompt ({len(prompt)} chars)...")

        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
            logger.info("✅ Streaming complete.")

        except Exception as e:
            logger.error(f"❌ Streaming generation failed: {e}")
            raise

    def generate_rag_answer(self, question: str, context: str) -> str:
        """Generate an answer using RAG-style prompt with question + context."""
        if not question or not context:
            raise ValueError("❌ Question and context must be non-empty strings.")

        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

        return self.generate_answer(prompt)

    def get_model_name(self) -> str:
        return self.model_name

    @classmethod
    def list_available_models(cls) -> dict:
        return cls.AVAILABLE_MODELS


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generator = GeminiGenerator(model_key="flash25", temperature=0.7)

    print("\n--- Model Info ---")
    print(f"Model : {generator.get_model_name()}")

    # Simple generation
    print("\n--- Simple Generation ---")
    answer = generator.generate_answer("What is LangChain in one sentence?")
    print(f"Answer: {answer}")

    # Streaming generation
    print("\n--- Streaming Generation ---")
    print("Streaming: ", end="", flush=True)
    for chunk in generator.generate_answer_stream("What is LangChain in one sentence?"):
        print(chunk, end="", flush=True)
    print()

    # RAG-style generation
    print("\n--- RAG Generation ---")
    context = """FastAPI is a modern, fast web framework for building APIs with Python.
    It is based on standard Python type hints. It is one of the fastest Python frameworks available."""
    question = "What is FastAPI?"
    answer = generator.generate_rag_answer(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer  : {answer}")

    print("\n--- Available Models ---")
    for key, name in GeminiGenerator.list_available_models().items():
        print(f"  {key:10} → {name}")
