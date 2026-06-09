import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".md": TextLoader,  # lightweight, no extra deps
    }

    # extensions that need encoding="utf-8"
    TEXT_EXTENSIONS = {".txt", ".md"}

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
        self.data_path = data_path

    def _load_file(self, file_path: str):
        """Load a single file and return its documents."""
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = self.SUPPORTED_EXTENSIONS.get(ext)

        if loader_class is None:
            return []

        try:
            if ext in self.TEXT_EXTENSIONS:
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)

            docs = loader.load()
            logger.info(f"✅ Loaded {len(docs)} page(s) from: {file_path}")
            return docs

        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            return []

    def load_documents(self):
        """Walk data_path and load all supported documents."""
        documents = []
        total_files = 0

        for root, _, files in os.walk(self.data_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue

                file_path = os.path.join(root, file)
                docs = self._load_file(file_path)
                documents.extend(docs)
                total_files += 1

        logger.info(f"📄 Total files loaded : {total_files}")
        logger.info(f"📄 Total pages/chunks : {len(documents)}")
        return documents

    def get_supported_extensions(self):
        """Return list of supported file extensions."""
        return list(self.SUPPORTED_EXTENSIONS.keys())


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DocumentLoader(data_path="data/")
    docs = loader.load_documents()

    print(f"\nLoaded {len(docs)} document chunks total.")

    if docs:
        print("\n--- First document preview ---")
        print(f"Source : {docs[0].metadata.get('source', 'N/A')}")
        print(f"Content: {docs[0].page_content[:300]}...")
