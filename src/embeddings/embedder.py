import logging
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    AVAILABLE_MODELS = {
        "txtsmall"     : "text-embedding-3-small",                   # OpenAI's text embedding
        "minilm"  : "sentence-transformers/all-MiniLM-L6-v2",    # fast, lightweight
        "mpnet"   : "sentence-transformers/all-mpnet-base-v2",    # better quality
        "bge"     : "BAAI/bge-base-en-v1.5",                      # best for RAG
    }

    def __init__(self, model_key: str = "minilm", device: str = "cpu"):
        self.model_key  = model_key
        self.device     = device
        self.model_name = self.AVAILABLE_MODELS.get(model_key, model_key)

        logger.info(f"🔄 Loading embedding model: {self.model_name}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name      = self.model_name,
            model_kwargs    = {"device": self.device},
            encode_kwargs   = {"normalize_embeddings": True},
        )

        logger.info(f"✅ Embedding model loaded: {self.model_name} on {self.device}")

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        return self.embedding_model

    def embed_text(self, text: str) -> list:
        """Embed a single string and return the vector."""
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: list) -> list:
        """Embed a list of strings and return list of vectors."""
        if not texts:
            raise ValueError("Input list is empty.")
        logger.info(f"🔢 Embedding {len(texts)} texts...")
        vectors = self.embedding_model.embed_documents(texts)
        logger.info(f"✅ Generated {len(vectors)} embeddings.")
        return vectors

    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        sample_vector = self.embed_text("test")
        return len(sample_vector)

    @classmethod
    def list_available_models(cls) -> dict:
        """Return all available model options."""
        return cls.AVAILABLE_MODELS


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generator = EmbeddingGenerator(model_key="minilm", device="cpu")

    print(f"\n--- Embedding Model Info ---")
    print(f"Model name : {generator.get_model_name()}")
    print(f"Dimension  : {generator.get_embedding_dimension()}")

    # Single text
    vector = generator.embed_text("What is LangChain?")
    print(f"Vector size: {len(vector)}")
    print(f"First 5 dims: {vector[:5]}")

    # Multiple texts
    texts   = ["LangChain is a framework.", "FastAPI is a web framework."]
    vectors = generator.embed_documents(texts)
    print(f"\nEmbedded {len(vectors)} documents.")

    print(f"\n--- Available Models ---")
    for key, name in EmbeddingGenerator.list_available_models().items():
        print(f"  {key:8} → {name}")