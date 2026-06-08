import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators work well for markdown + plain text
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        if not documents:
            logger.warning("⚠️  No documents provided to chunk.")
            return []

        chunks = self.splitter.split_documents(documents)

        logger.info(f"📥 Input documents : {len(documents)}")
        logger.info(f"📤 Output chunks   : {len(chunks)}")
        logger.info(f"📐 Chunk size      : {self.chunk_size}")
        logger.info(f"🔁 Chunk overlap   : {self.chunk_overlap}")

        return chunks

    def get_chunk_stats(self, chunks: list) -> dict:
        """Return basic stats about the chunks."""
        if not chunks:
            return {}

        lengths = [len(chunk.page_content) for chunk in chunks]
        return {
            "total_chunks"  : len(chunks),
            "avg_length"    : round(sum(lengths) / len(lengths), 2),
            "min_length"    : min(lengths),
            "max_length"    : max(lengths),
        }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
                         "It provides tools for chaining together different components. "
                         "You can build RAG pipelines, agents, and chatbots with it. " * 20,
            metadata={"source": "langchain.md"}
        ),
        Document(
            page_content="FastAPI is a modern, fast web framework for building APIs with Python. "
                         "It is based on standard Python type hints and is very easy to use. " * 20,
            metadata={"source": "fastapi.md"}
        ),
    ]

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(sample_docs)

    stats = chunker.get_chunk_stats(chunks)
    print(f"\n--- Chunk Stats ---")
    for key, value in stats.items():
        print(f"  {key:15}: {value}")

    print(f"\n--- First Chunk Preview ---")
    print(f"Source : {chunks[0].metadata.get('source')}")
    print(f"Content: {chunks[0].page_content[:300]}...")