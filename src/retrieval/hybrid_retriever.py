import logging
from src.embeddings.embedder import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        model_key: str = "minilm",
        device: str = "cpu",
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        final_top_k: int = 20,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        logger.info("🚀 Initializing Hybrid Retriever...")

        # Dense retriever (FAISS)
        embedder = EmbeddingGenerator(model_key=model_key, device=device)
        self.vector_store_manager = VectorStoreManager(
            embedding_model=embedder.get_embedding_model()
        )
        self.vector_store = self.vector_store_manager.load_vector_store()

        # Sparse retriever (BM25)
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.load()

        logger.info("✅ Hybrid Retriever ready.")
        logger.info(f"   Dense weight : {self.dense_weight}")
        logger.info(f"   Sparse weight: {self.sparse_weight}")

    def _get_dense_results(self, query: str, top_k: int) -> list:
        """Retrieve using FAISS dense embeddings."""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        # results = [(doc, score), ...] — lower score = more similar in FAISS
        return results

    def _get_sparse_results(self, query: str, top_k: int) -> list:
        """Retrieve using BM25 sparse retrieval."""
        docs = self.bm25_retriever.retrieve(query, top_k=top_k)
        # Assign fake scores 1.0 → 0.0 descending for merging
        scores = [1.0 - (i / len(docs)) for i in range(len(docs))]
        return list(zip(docs, scores))

    def _reciprocal_rank_fusion(
        self, dense_results: list, sparse_results: list, k: int = 60
    ) -> list:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
        RRF score = Σ 1 / (k + rank)
        Higher score = more relevant.
        """
        scores = {}  # content → fused score
        docs = {}  # content → Document object

        # Score dense results
        for rank, (doc, _) in enumerate(dense_results, 1):
            content = doc.page_content
            docs[content] = doc
            scores[content] = scores.get(content, 0) + (
                self.dense_weight * (1 / (k + rank))
            )

        # Score sparse results
        for rank, (doc, _) in enumerate(sparse_results, 1):
            content = doc.page_content
            docs[content] = doc
            scores[content] = scores.get(content, 0) + (
                self.sparse_weight * (1 / (k + rank))
            )

        # Sort by fused score descending
        sorted_content = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        return [docs[c] for c in sorted_content]

    def retrieve(self, query: str, top_k: int = None) -> list:
        """Run hybrid retrieval and return top_k fused results."""
        if not query:
            raise ValueError("❌ Query must be a non-empty string.")

        k = top_k or self.final_top_k

        logger.info(f"🔍 Hybrid retrieval for: '{query}' (top_k={k})")

        # Dense + Sparse
        dense_results = self._get_dense_results(query, self.dense_top_k)
        sparse_results = self._get_sparse_results(query, self.sparse_top_k)

        logger.info(f"   Dense  results : {len(dense_results)}")
        logger.info(f"   Sparse results : {len(sparse_results)}")

        # Fuse with RRF
        fused_docs = self._reciprocal_rank_fusion(dense_results, sparse_results)
        final_docs = fused_docs[:k]

        logger.info(f"✅ Hybrid retrieval complete → {len(final_docs)} docs.")
        return final_docs


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever(dense_weight=0.6, sparse_weight=0.4, final_top_k=5)

    query = "What is FastAPI?"
    results = retriever.retrieve(query, top_k=5)

    print("\n--- Hybrid Retrieval Results ---")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Source : {doc.metadata.get('source', 'N/A')}")
        print(f"     Content: {doc.page_content[:200]}...")
