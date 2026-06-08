import os
import json
import shutil
import logging
import time
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_DIR        = "data/uploads"
PROCESSED_DIR     = "data/processed"
CHUNKS_PATH       = "data/processed/chunks.json"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
SUPPORTED_TYPES   = {".pdf", ".txt", ".docx", ".md"}


class UploadPipeline:

    def __init__(self):
        logger.info("🚀 Initializing Upload Pipeline...")

        self.cleaner  = TextCleaner()
        self.chunker  = DocumentChunker(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )

        # Load embedding model
        embedder = EmbeddingGenerator(model_key="minilm", device="cpu")
        self.embedding_model = embedder.get_embedding_model()
        self.embedder        = embedder

        # Load vector store manager
        self.vector_store_manager = VectorStoreManager(
            embedding_model = self.embedding_model
        )

        # Load BM25
        self.bm25 = BM25Retriever()

        # Load existing vector store if available
        if self.vector_store_manager.index_exists():
            self.vector_store = self.vector_store_manager.load_vector_store()
            logger.info("📂 Existing FAISS index loaded.")
        else:
            self.vector_store = None
            logger.info("⚠️  No existing FAISS index — will create on first upload.")

        self.tracker = MLflowTracker()
        os.makedirs(UPLOAD_DIR,    exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        logger.info("✅ Upload Pipeline ready.")

    def _save_uploaded_file(self, file_name: str, file_bytes: bytes) -> str:
        """Save uploaded file bytes to the uploads directory."""
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_TYPES:
            raise ValueError(
                f"❌ Unsupported file type: '{ext}'. "
                f"Supported: {SUPPORTED_TYPES}"
            )

        dest_path = os.path.join(UPLOAD_DIR, file_name)
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"💾 File saved to: {dest_path}")
        return dest_path

    def _load_existing_chunks(self) -> list:
        """Load existing chunks from JSON."""
        if not os.path.exists(CHUNKS_PATH):
            return []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content = item["content"],
                metadata     = item["metadata"]
            )
            for item in data
        ]

    def _save_chunks(self, chunks: list) -> None:
        """Serialize and save all chunks to JSON."""
        serialized = [
            {
                "content" : chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4, ensure_ascii=False)

        logger.info(f"💾 {len(chunks)} total chunks saved to: {CHUNKS_PATH}")

    def _rebuild_bm25(self, all_chunks: list) -> None:
        """Rebuild BM25 index from all chunks."""
        self.bm25.build(all_chunks)
        self.bm25.save()
        logger.info(f"✅ BM25 index rebuilt with {len(all_chunks)} chunks.")

    def process(self, file_name: str, file_bytes: bytes) -> dict:
        """
        Full upload pipeline:
        save → load → clean → chunk → update FAISS → rebuild BM25 → save chunks → track
        """
        logger.info(f"📄 Processing upload: {file_name}")

        with self.tracker.start_run(f"upload_{file_name}"):
            pipeline_start = time.time()

            # --- Step 1: Save file ---
            logger.info("💾 Step 1: Saving uploaded file...")
            file_path = self._save_uploaded_file(file_name, file_bytes)

            # --- Step 2: Load document ---
            logger.info("📂 Step 2: Loading document...")
            loader    = DocumentLoader(UPLOAD_DIR)
            documents = loader.load_documents()

            # Filter to only the newly uploaded file
            new_docs = [
                doc for doc in documents
                if file_name in doc.metadata.get("source", "")
            ]

            if not new_docs:
                raise ValueError(f"❌ Could not load document: {file_name}")

            logger.info(f"   Loaded {len(new_docs)} page(s) from: {file_name}")

            # --- Step 3: Clean ---
            logger.info("🧹 Step 3: Cleaning documents...")
            cleaned_docs = self.cleaner.clean_documents(new_docs)

            # --- Step 4: Chunk ---
            logger.info("✂️  Step 4: Chunking documents...")
            new_chunks = self.chunker.split_documents(cleaned_docs)
            logger.info(f"   Created {len(new_chunks)} new chunks.")

            # --- Step 5: Update FAISS ---
            logger.info("🔢 Step 5: Updating FAISS index...")
            embed_start = time.time()

            if self.vector_store is None:
                # First upload — create index from scratch
                self.vector_store = self.vector_store_manager.create_vector_store(
                    new_chunks
                )
            else:
                # Subsequent uploads — add to existing index
                self.vector_store = self.vector_store_manager.add_documents(
                    self.vector_store,
                    new_chunks
                )

            embed_latency = round(time.time() - embed_start, 4)
            self.vector_store_manager.save_vector_store(self.vector_store)
            logger.info(f"   FAISS updated in {embed_latency}s.")

            # --- Step 6: Update BM25 ---
            logger.info("📦 Step 6: Rebuilding BM25 index...")
            existing_chunks = self._load_existing_chunks()
            all_chunks      = existing_chunks + new_chunks
            self._rebuild_bm25(all_chunks)

            # --- Step 7: Save all chunks ---
            logger.info("💾 Step 7: Saving updated chunks...")
            self._save_chunks(all_chunks)

            total_latency = round(time.time() - pipeline_start, 4)

            # --- Step 8: Track ---
            logger.info("📊 Step 8: Logging to MLflow...")
            self.tracker.log_rag_parameters(
                chunk_size      = CHUNK_SIZE,
                chunk_overlap   = CHUNK_OVERLAP,
                embedding_model = self.embedder.get_model_name(),
                top_k           = 0,
                vector_db       = "FAISS + BM25",
                llm_model       = "not_applicable"
            )
            self.tracker.log_metric("new_pages",       len(new_docs))
            self.tracker.log_metric("new_chunks",      len(new_chunks))
            self.tracker.log_metric("total_chunks",    len(all_chunks))
            self.tracker.log_metric("embed_latency",   embed_latency)
            self.tracker.log_metric("total_latency",   total_latency)

            result = {
                "file_name"    : file_name,
                "new_pages"    : len(new_docs),
                "new_chunks"   : len(new_chunks),
                "total_chunks" : len(all_chunks),
                "embed_latency": embed_latency,
                "total_latency": total_latency,
                "status"       : "success",
            }

            logger.info("✅ Upload pipeline complete.")
            logger.info(f"   File          : {file_name}")
            logger.info(f"   New pages     : {len(new_docs)}")
            logger.info(f"   New chunks    : {len(new_chunks)}")
            logger.info(f"   Total chunks  : {len(all_chunks)}")
            logger.info(f"   Embed latency : {embed_latency}s")
            logger.info(f"   Total latency : {total_latency}s")

            return result

    def delete_document(self, file_name: str) -> dict:
        """
        Remove a document from uploads, chunks, FAISS, and BM25.
        """
        logger.info(f"🗑️  Deleting document: {file_name}")

        # --- Remove file from disk ---
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"   File deleted: {file_path}")

        # --- Remove chunks belonging to this file ---
        existing_chunks  = self._load_existing_chunks()
        filtered_chunks  = [
            c for c in existing_chunks
            if file_name not in c.metadata.get("source", "")
        ]
        removed_count    = len(existing_chunks) - len(filtered_chunks)
        logger.info(f"   Removed {removed_count} chunks for: {file_name}")

        # --- Save updated chunks ---
        self._save_chunks(filtered_chunks)

        # --- Rebuild FAISS from scratch ---
        if filtered_chunks:
            logger.info("🔢 Rebuilding FAISS index without deleted document...")
            self.vector_store = self.vector_store_manager.create_vector_store(
                filtered_chunks
            )
            self.vector_store_manager.save_vector_store(self.vector_store)
        else:
            logger.warning("⚠️  No chunks remaining — FAISS index cleared.")
            self.vector_store = None

        # --- Rebuild BM25 ---
        self._rebuild_bm25(filtered_chunks)

        return {
            "file_name"      : file_name,
            "chunks_removed" : removed_count,
            "chunks_remaining": len(filtered_chunks),
            "status"         : "deleted",
        }

    def list_uploaded_files(self) -> list:
        """List all uploaded files with metadata."""
        files = []
        for f in os.listdir(UPLOAD_DIR):
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_TYPES:
                path = os.path.join(UPLOAD_DIR, f)
                files.append({
                    "file_name": f,
                    "size_kb"  : round(os.path.getsize(path) / 1024, 2),
                    "extension": ext,
                })
        return files


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = UploadPipeline()

    # List existing files
    print("\n--- Uploaded Files ---")
    files = pipeline.list_uploaded_files()
    if files:
        for f in files:
            print(f"  {f['file_name']} ({f['size_kb']} KB)")
    else:
        print("  No files uploaded yet.")

    # Simulate uploading a text file
    test_file    = "test_doc.txt"
    test_content = b"""FastAPI is a modern Python web framework.
It is very fast and easy to use.
It supports async out of the box.
It is based on standard Python type hints."""

    print(f"\n--- Processing Upload: {test_file} ---")
    result = pipeline.process(test_file, test_content)
    print(f"\nResult:")
    for key, value in result.items():
        print(f"  {key:15}: {value}")