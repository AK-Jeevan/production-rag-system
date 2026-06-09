import logging
from src.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        model_key: str = "minilm",
        device: str = "cpu",
        top_k: int = 20,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.top_k = top_k

        logger.info("🔄 Loading Hybrid Retriever...")
        self.hybrid_retriever = HybridRetriever(
            model_key=model_key,
            device=device,
            final_top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        logger.info("✅ Retriever ready.")

    def retrieve(self, query: str, top_k: int = None) -> list:
        """Retrieve top_k most relevant chunks using hybrid search."""
        if not query or not isinstance(query, str):
            raise ValueError("❌ Query must be a non-empty string.")

        k = top_k or self.top_k
        logger.info(f"🔍 Retrieving top {k} chunks for: '{query}'")
        results = self.hybrid_retriever.retrieve(query, top_k=k)
        logger.info(f"✅ Retrieved {len(results)} chunks.")
        return results

    def retrieve_with_scores(self, query: str, top_k: int = None) -> list:
        """Retrieve chunks with descending RRF scores."""
        if not query or not isinstance(query, str):
            raise ValueError("❌ Query must be a non-empty string.")

        k = top_k or self.top_k
        results = self.retrieve(query, top_k=k)

        # Assign descending scores for compatibility with downstream code
        scored = [
            (doc, round(1.0 - (i / len(results)), 4)) for i, doc in enumerate(results)
        ]
        logger.info(f"✅ Retrieved {len(scored)} chunks with scores.")
        return scored

    def format_context(self, results: list) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            context_parts.append(f"[{i}] Source: {source}\n{content}")

        return "\n\n".join(context_parts)

    def retrieve_and_format(self, query: str, top_k: int = None) -> tuple:
        """Retrieve chunks and return both raw results and formatted context string."""
        results = self.retrieve(query, top_k=top_k)
        context = self.format_context(results)
        return results, context


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = Retriever(
        model_key="minilm",
        device="cpu",
        top_k=20,
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    query = "What is FastAPI?"

    # Basic retrieval
    print("\n--- Basic Retrieval ---")
    results = retriever.retrieve(query, top_k=5)
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Source : {doc.metadata.get('source', 'N/A')}")
        print(f"     Content: {doc.page_content[:300]}...")

    # Retrieval with scores
    print("\n--- Retrieval with Scores ---")
    scored_results = retriever.retrieve_with_scores(query, top_k=5)
    for doc, score in scored_results:
        print(f"\nScore  : {score}")
        print(f"Source : {doc.metadata.get('source', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")

    # Retrieve and format
    print("\n--- Formatted Context ---")
    results, context = retriever.retrieve_and_format(query, top_k=3)
    print(context)
