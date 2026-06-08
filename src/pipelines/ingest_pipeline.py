import os
import json
import logging
from src.ingestion.loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker
from src.utils.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw"
PROCESSED_DATA_PATH = "data/processed/chunks.json"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200


def save_chunks(chunks: list) -> None:
    """Serialize and save chunks to JSON."""
    serialized_chunks = [
        {
            "content" : chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]

    os.makedirs("data/processed", exist_ok=True)

    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized_chunks, f, indent=4, ensure_ascii=False)

    logger.info(f"💾 Saved {len(chunks)} chunks to: {PROCESSED_DATA_PATH}")


def run_ingestion() -> None:
    """Full ingestion pipeline: load → clean → chunk → save → track."""
    logger.info("🚀 Starting ingestion pipeline...")

    tracker = MLflowTracker()

    with tracker.start_run("document_ingestion_pipeline"):

        # --- Step 1: Load ---
        logger.info("📂 Step 1: Loading documents...")
        loader    = DocumentLoader(RAW_DATA_PATH)
        documents = loader.load_documents()

        if not documents:
            logger.error("❌ No documents found. Check your data/raw folder.")
            return

        # --- Step 2: Clean ---
        logger.info("🧹 Step 2: Cleaning documents...")
        cleaner           = TextCleaner()
        cleaned_documents = cleaner.clean_documents(documents)

        # --- Step 3: Chunk ---
        logger.info("✂️  Step 3: Chunking documents...")
        chunker = DocumentChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = chunker.split_documents(cleaned_documents)
        stats  = chunker.get_chunk_stats(chunks)

        # --- Step 4: Save ---
        logger.info("💾 Step 4: Saving chunks...")
        save_chunks(chunks)

        # --- Step 5: Track ---
        logger.info("📊 Step 5: Logging metrics to MLflow...")
        tracker.log_rag_parameters(
            chunk_size      = CHUNK_SIZE,
            chunk_overlap   = CHUNK_OVERLAP,
            embedding_model = "not_yet_set",
            top_k           = 0,
            vector_db       = "not_yet_set",
            llm_model       = "not_yet_set"
        )

        # use log_metric not log_metrics (matches your MLflowTracker API)
        mlflow_metrics = {
            "documents_loaded" : len(documents),
            "documents_cleaned": len(cleaned_documents),
            "chunks_created"   : len(chunks),
            "avg_chunk_length" : stats.get("avg_length", 0),
            "min_chunk_length" : stats.get("min_length", 0),
            "max_chunk_length" : stats.get("max_length", 0),
        }

        for key, value in mlflow_metrics.items():
            tracker.log_metric(key, value)

        # --- Summary ---
        logger.info("✅ Ingestion pipeline complete.")
        logger.info(f"   Documents loaded  : {len(documents)}")
        logger.info(f"   Documents cleaned : {len(cleaned_documents)}")
        logger.info(f"   Chunks created    : {len(chunks)}")
        logger.info(f"   Avg chunk length  : {stats.get('avg_length', 0)}")
        logger.info(f"   Saved to          : {PROCESSED_DATA_PATH}")
        logger.info("   Check http://127.0.0.1:5000 for MLflow run.")


if __name__ == "__main__":
    run_ingestion()