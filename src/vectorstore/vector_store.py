import os
import logging
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, embedding_model, index_path: str = "models/faiss_index"):
        self.embedding_model = embedding_model
        self.index_path = index_path

    def create_vector_store(self, chunks: list) -> FAISS:
        """Create a FAISS vector store from document chunks."""
        if not chunks:
            raise ValueError("❌ No chunks provided to create vector store.")

        logger.info(f"🔄 Creating FAISS vector store from {len(chunks)} chunks...")

        vector_store = FAISS.from_documents(
            documents=chunks, embedding=self.embedding_model
        )

        logger.info(f"✅ Vector store created with {len(chunks)} vectors.")
        return vector_store

    def save_vector_store(self, vector_store: FAISS) -> None:
        """Save FAISS index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        vector_store.save_local(self.index_path)
        logger.info(f"💾 Vector store saved to: {self.index_path}")

    def load_vector_store(self) -> FAISS:
        """Load FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"❌ No FAISS index found at: {self.index_path}. "
                f"Run create_vector_store() first."
            )

        logger.info(f"📂 Loading FAISS index from: {self.index_path}")

        vector_store = FAISS.load_local(
            self.index_path, self.embedding_model, allow_dangerous_deserialization=True
        )

        logger.info("✅ Vector store loaded successfully.")
        return vector_store

    def index_exists(self) -> bool:
        """Check if a saved FAISS index exists on disk."""
        return os.path.exists(self.index_path)

    def get_or_create(self, chunks: list) -> FAISS:
        """Load existing index if available, otherwise create and save a new one."""
        if self.index_exists():
            logger.info("📂 Existing index found — loading from disk.")
            return self.load_vector_store()

        logger.info("🆕 No existing index — creating from chunks.")
        vector_store = self.create_vector_store(chunks)
        self.save_vector_store(vector_store)
        return vector_store

    def add_documents(self, vector_store: FAISS, chunks: list) -> FAISS:
        """Add new chunks to an existing vector store and save."""
        if not chunks:
            raise ValueError("❌ No chunks provided to add.")

        logger.info(f"➕ Adding {len(chunks)} new chunks to vector store...")
        vector_store.add_documents(chunks)
        self.save_vector_store(vector_store)
        logger.info("✅ New chunks added and index saved.")
        return vector_store

    def similarity_search(
        self, vector_store: FAISS, query: str, top_k: int = 5
    ) -> list:
        """Run a similarity search and return top_k results."""
        if not query:
            raise ValueError("❌ Query string is empty.")

        logger.info(f"🔍 Searching for: '{query}' (top_k={top_k})")
        results = vector_store.similarity_search(query, k=top_k)
        logger.info(f"✅ Found {len(results)} results.")
        return results


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.embeddings.embedder import EmbeddingGenerator

    # Load embedding model
    generator = EmbeddingGenerator(model_key="minilm", device="cpu")
    embedding = generator.get_embedding_model()

    # Init manager
    manager = VectorStoreManager(embedding_model=embedding)

    # Create dummy chunks
    from langchain_core.documents import Document

    chunks = [
        Document(
            page_content="LangChain is a framework for LLM applications.",
            metadata={"source": "langchain.md"},
        ),
        Document(
            page_content="FastAPI is a modern Python web framework.",
            metadata={"source": "fastapi.md"},
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"source": "faiss.md"},
        ),
        Document(
            page_content="MLflow tracks machine learning experiments.",
            metadata={"source": "mlflow.md"},
        ),
        Document(
            page_content="DVC is used for data version control in ML.",
            metadata={"source": "dvc.md"},
        ),
    ]

    # Create and save
    vs = manager.get_or_create(chunks)

    # Search
    results = manager.similarity_search(vs, query="What is LangChain?", top_k=2)
    print("\n--- Search Results ---")
    for i, doc in enumerate(results):
        print(f"\n[{i + 1}] Source : {doc.metadata.get('source')}")
        print(f"     Content: {doc.page_content}")
