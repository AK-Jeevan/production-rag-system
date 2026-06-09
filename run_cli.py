import sys
import logging
from src.pipelines.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_separator(char: str = "=", length: int = 80) -> None:
    print(char * length)


def print_response(result: dict) -> None:
    print_separator()
    print(f"❓ Question          : {result.get('question', 'N/A')}")
    print(f"✏️  Rewritten Query   : {result.get('rewritten_query', 'N/A')}")
    print_separator("-")
    print(f"💬 Answer            :\n\n{result.get('answer', 'N/A')}")
    print_separator("-")
    print("📚 Sources           :")
    for source in result.get("sources", []):
        print(f"   • {source}")
    print_separator("-")
    print(f"📄 Docs Retrieved    : {len(result.get('retrieved_docs', []))}")
    print(f"⏱️  Retrieval Latency : {result.get('retrieval_latency', 0.0)}s")
    print(f"⏱️  Rerank Latency    : {result.get('rerank_latency', 0.0)}s")
    print(f"⏱️  Generation Latency: {result.get('generation_latency', 0.0)}s")
    print(f"⏱️  Total Latency     : {result.get('total_latency', 0.0)}s")
    print(f"🔢 Input Tokens      : {result.get('input_tokens', 0)}")
    print(f"🔢 Output Tokens     : {result.get('output_tokens', 0)}")
    print(f"💰 Estimated Cost    : ${result.get('estimated_cost', 0.0)}")
    print(f"🧠 Memory Messages   : {result.get('memory_messages', 0)}")
    print_separator()


def run_interactive() -> None:
    """Run the RAG pipeline in interactive mode."""
    logger.info("🚀 Starting RAG Assistant...")

    try:
        rag = RAGPipeline(
            retrieval_top_k=20, rerank_top_k=5, model_key="flash25", temperature=0.7
        )
        logger.info("✅ RAG Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Pipeline: {e}")
        sys.exit(1)

    print_separator()
    print("  🤖 RAG Assistant — Powered by Gemini + FAISS + Reranker + LangChain")
    print("  Type 'exit', 'quit', or 'q' to stop.")
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
