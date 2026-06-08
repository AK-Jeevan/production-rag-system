import sys
import logging
from src.pipelines.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_separator(char: str = "=", length: int = 80) -> None:
    print(char * length)


def print_response(result: dict) -> None:
    """Pretty print the RAG pipeline response."""
    print_separator()
    print(f"❓ Question          : {result['question']}")
    print_separator("-")
    print(f"💬 Answer            :\n\n{result['answer']}")
    print_separator("-")
    print(f"📄 Docs Retrieved    : {result['retrieved_docs']}")
    print(f"⏱️  Retrieval Latency : {result['retrieval_latency']}s")
    print(f"⏱️  Generation Latency: {result['generation_latency']}s")
    print(f"⏱️  Total Latency     : {result['total_latency']}s")
    print(f"🔢 Approx Tokens     : {result['prompt_tokens']}")
    print_separator()


def run_interactive() -> None:
    """Run the RAG pipeline in interactive mode."""
    logger.info("🚀 Starting RAG Assistant...")

    try:
        rag = RAGPipeline(
            top_k       = 5,
            model_key   = "flash25",
            temperature = 0.7
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Pipeline: {e}")
        sys.exit(1)

    print_separator()
    print("  🤖 RAG Assistant — Powered by Gemini + FAISS + LangChain")
    print("  Type 'exit' or 'quit' to stop.")
    print_separator()

    while True:
        try:
            question = input("\n❓ Ask your question: ").strip()

            # Exit commands
            if question.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!")
                break

            # Empty input
            if not question:
                print("⚠️  Please enter a question.")
                continue

            # Run pipeline
            result = rag.ask(question)
            print_response(result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            print(f"\n❌ Something went wrong: {e}")
            print("   Please try again.\n")
            continue


if __name__ == "__main__":
    run_interactive()