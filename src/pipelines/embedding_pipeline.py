import os
import json
import time
import logging

from langchain_core.documents import Document

from src.embeddings.embedder import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_PATH   = "data/processed/chunks.json"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


def load_chunks() -> list:
    """Load chunks from processed JSON file."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"❌ Chunks file not found at: {CHUNKS_PATH}. "
            f"Run the ingestion pipeline first."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("❌ Chunks file is empty.")

    chunks = [
        Document(
            page_content = item["content"],
            metadata     = item["metadata"]
        )
        for item in data
    ]

    logger.info(f"📂 Loaded {len(chunks)} chunks from: {CHUNKS_PATH}")
    return chunks


def run_embedding_pipeline() -> None:
    """Full embedding pipeline: load chunks → embed → save FAISS → build BM25 → track."""
    logger.info("🚀 Starting embedding pipeline...")

    tracker = MLflowTracker()

    with tracker.start_run("embedding_pipeline"):

        # --- Step 1: Load chunks ---
        logger.info("📂 Step 1: Loading chunks...")
        chunks = load_chunks()

        # --- Step 2: Load embedding model ---
        logger.info("🔄 Step 2: Loading embedding model...")
        embedder = EmbeddingGenerator(model_key="minilm", device="cpu")

        tracker.log_rag_parameters(
            chunk_size      = CHUNK_SIZE,
            chunk_overlap   = CHUNK_OVERLAP,
            embedding_model = embedder.get_model_name(),
            top_k           = 0,
            vector_db       = "FAISS + BM25",
            llm_model       = "not_yet_set"
        )

        # --- Step 3: Create FAISS vector store ---
        logger.info("🔢 Step 3: Embedding chunks and creating FAISS index...")
        vector_store_manager = VectorStoreManager(
            embedding_model = embedder.get_embedding_model()
        )

        embedding_start   = time.time()
        vector_store      = vector_store_manager.create_vector_store(chunks)
        embedding_latency = round(time.time() - embedding_start, 4)

        # --- Step 4: Save FAISS index ---
        logger.info("💾 Step 4: Saving FAISS index...")
        vector_store_manager.save_vector_store(vector_store)
        logger.info(f"   FAISS index saved to : models/faiss_index")

        # --- Step 4b: Build and save BM25 index ---
        logger.info("📦 Step 4b: Building BM25 index...")
        bm25_start = time.time()

        bm25 = BM25Retriever()
        bm25.build(chunks)
        bm25.save()

        bm25_latency = round(time.time() - bm25_start, 4)
        logger.info(f"   BM25 index saved to  : models/bm25_index")

        # --- Step 5: Track metrics ---
        logger.info("📊 Step 5: Logging metrics to MLflow...")
        tracker.log_metric("chunks_embedded",   len(chunks))
        tracker.log_metric("embedding_latency", embedding_latency)
        tracker.log_metric("embedding_dim",     embedder.get_embedding_dimension())
        tracker.log_metric("bm25_latency",      bm25_latency)
        tracker.log_metric("bm25_index_built",  1)

        # --- Summary ---
        logger.info("✅ Embedding pipeline complete.")
        logger.info(f"   Chunks embedded   : {len(chunks)}")
        logger.info(f"   Embedding latency : {embedding_latency}s")
        logger.info(f"   Embedding dim     : {embedder.get_embedding_dimension()}")
        logger.info(f"   BM25 latency      : {bm25_latency}s")
        logger.info(f"   FAISS index       : models/faiss_index")
        logger.info(f"   BM25  index       : models/bm25_index")
        logger.info("   Check http://127.0.0.1:5000 for MLflow run.")


if __name__ == "__main__":
    run_embedding_pipeline()