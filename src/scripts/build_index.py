import os
import logging
from src.ingestion.loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStoreManager
from src.retrieval.bm25_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/models/faiss_index")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/models/bm25_index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY", "minilm")


def build_indices(force_rebuild: bool = False) -> None:
    faiss_exists = os.path.isdir(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH)
    bm25_exists = os.path.isfile(os.path.join(BM25_INDEX_PATH, "bm25.pkl"))

    if faiss_exists and bm25_exists and not force_rebuild:
        logger.info("✅ FAISS and BM25 indices already exist. Use force_rebuild=True to overwrite.")
        return

    if not os.path.isdir(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data path does not exist: {RAW_DATA_PATH}")

    logger.info("📂 Step 1: Loading documents...")
    loader = DocumentLoader(RAW_DATA_PATH)
    documents = loader.load_documents()
    if not documents:
        raise ValueError("❌ No documents found. Check your data/raw folder.")
    logger.info(f"   Loaded {len(documents)} documents.")

    logger.info("🧹 Step 2: Cleaning documents...")
    cleaner = TextCleaner()
    cleaned_documents = cleaner.clean_documents(documents)
    if not cleaned_documents:
        raise ValueError("❌ No documents remained after cleaning.")
    logger.info(f"   Cleaned documents: {len(cleaned_documents)}")

    logger.info("✂️  Step 3: Chunking documents...")
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.split_documents(cleaned_documents)
    if not chunks:
        raise ValueError("❌ No chunks created from documents.")
    logger.info(f"   Created {len(chunks)} chunks.")

    logger.info("🔢 Step 4: Loading embedding model...")
    embedder = EmbeddingGenerator(model_key=EMBEDDING_MODEL_KEY, device="cpu")
    embedding_model = embedder.get_embedding_model()

    if not faiss_exists or force_rebuild:
        logger.info("💾 Step 5a: Building FAISS index...")
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vs_manager = VectorStoreManager(embedding_model=embedding_model, index_path=FAISS_INDEX_PATH)
        faiss_index = vs_manager.create_vector_store(chunks)
        vs_manager.save_vector_store(faiss_index)
        logger.info(f"   FAISS index saved to: {FAISS_INDEX_PATH}")
    else:
        logger.info("⏭️  Step 5a: FAISS index already exists, skipping.")

    if not bm25_exists or force_rebuild:
        logger.info("💾 Step 5b: Building BM25 index...")
        os.makedirs(BM25_INDEX_PATH, exist_ok=True)
        bm25_retriever = BM25Retriever(save_path=BM25_INDEX_PATH)
        bm25_retriever.build(chunks)
        bm25_retriever.save()
        logger.info(f"   BM25 index saved to: {BM25_INDEX_PATH}")
    else:
        logger.info("⏭️  Step 5b: BM25 index already exists, skipping.")

    logger.info("✅ Index build complete.")


if __name__ == "__main__":
    build_indices(force_rebuild=False)
