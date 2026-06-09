import logging
from typing import Optional
from src.pipelines.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        model_key: str = "flash25",
        temperature: float = 0.7,
    ):
        try:
            self.pipeline = RAGPipeline(
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
                model_key=model_key,
                temperature=temperature,
            )
            logger.info("✅ RAGService initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAGService: {e}")
            raise

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        try:
            logger.info(f"🔍 Processing query: {question!r}")
            result = self.pipeline.ask(question)
            logger.info("✅ Query processed successfully.")
            return result
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
