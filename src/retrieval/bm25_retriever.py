import logging
import pickle
import os
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, save_path: str = "models/bm25_index"):
        self.save_path = save_path
        self.bm25 = None
        self.docs = []

    def build(self, documents: list) -> None:
        """Build BM25 index from documents."""
        if not documents:
            raise ValueError("❌ No documents provided to build BM25 index.")

        self.docs = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"✅ BM25 index built with {len(documents)} documents.")

    def save(self) -> None:
        """Save BM25 index and documents to disk."""
        if self.bm25 is None:
            raise ValueError("❌ BM25 index not built yet. Run build() first.")

        os.makedirs(self.save_path, exist_ok=True)

        with open(f"{self.save_path}/bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        with open(f"{self.save_path}/docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)

        logger.info(f"💾 BM25 index saved to: {self.save_path}")

    def load(self) -> None:
        """Load BM25 index and documents from disk."""
        bm25_path = f"{self.save_path}/bm25.pkl"
        docs_path = f"{self.save_path}/docs.pkl"

        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"❌ No BM25 index found at: {bm25_path}. Run build() and save() first."
            )

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)

        logger.info(f"📂 BM25 index loaded from: {self.save_path}")
        logger.info(f"   Documents: {len(self.docs)}")

    def retrieve(self, query: str, top_k: int = 20) -> list:
        """Retrieve top_k documents using BM25 scoring."""
        if self.bm25 is None:
            raise ValueError("❌ BM25 index not loaded. Run build() or load() first.")

        if not query:
            raise ValueError("❌ Query must be a non-empty string.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = [self.docs[i] for i in top_indices]
        logger.info(f"✅ BM25 retrieved {len(results)} docs for: '{query}'")
        return results

    def index_exists(self) -> bool:
        return os.path.exists(f"{self.save_path}/bm25.pkl")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="FastAPI is a modern Python web framework.",
            metadata={"source": "fastapi.md"},
        ),
        Document(
            page_content="LangChain is a framework for LLM applications.",
            metadata={"source": "langchain.md"},
        ),
        Document(
            page_content="MLflow tracks machine learning experiments.",
            metadata={"source": "mlflow.md"},
        ),
        Document(
            page_content="FAISS is used for similarity search.",
            metadata={"source": "faiss.md"},
        ),
        Document(
            page_content="DVC versions data and ML pipelines.",
            metadata={"source": "dvc.md"},
        ),
    ]

    retriever = BM25Retriever()
    retriever.build(docs)
    retriever.save()

    retriever2 = BM25Retriever()
    retriever2.load()

    results = retriever2.retrieve("What is FastAPI?", top_k=3)
    print("\n--- BM25 Results ---")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('source')} → {doc.page_content}")
