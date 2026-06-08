import logging
from src.embeddings.embedder import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_similarity_search(query: str, top_k: int = 3) -> list:
    """Load FAISS index and run a similarity search."""

    # --- Step 1: Load embedding model ---
    logger.info("🔄 Loading embedding model...")
    embedder = EmbeddingGenerator(model_key="minilm", device="cpu")

    # --- Step 2: Load vector store ---
    logger.info("📂 Loading FAISS index...")
    vector_store_manager = VectorStoreManager(
        embedding_model=embedder.get_embedding_model()
    )
    vector_store = vector_store_manager.load_vector_store()

    # --- Step 3: Search ---
    logger.info(f"🔍 Searching for: '{query}' (top_k={top_k})")
    results = vector_store.similarity_search(query, k=top_k)

    # --- Step 4: Display ---
    print(f"\n{'='*80}")
    print(f"  Query  : {query}")
    print(f"  Top K  : {top_k}")
    print(f"{'='*80}\n")

    if not results:
        print("⚠️  No results found.")
        return []

    for i, result in enumerate(results, 1):
        print(f"[{i}] Source : {result.metadata.get('source', 'N/A')}")
        print(f"    Content: {result.page_content[:300]}...")
        print("=" * 80)

    return results


if __name__ == "__main__":
    queries = [
        "What is FastAPI?"
    ]

    for query in queries:
        run_similarity_search(query=query, top_k=3)
        print("\n")