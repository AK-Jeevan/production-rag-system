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
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.model_key = model_key
        self.temperature = temperature
        self.pipeline: Optional[RAGPipeline] = None
        logger.info("✅ RAGService instance created (pipeline not yet initialized).")

    def get_pipeline(self) -> RAGPipeline:
        """Eagerly initialize and return the RAG pipeline."""
        if self.pipeline is None:
            try:
                logger.info("🔄 Initializing RAG pipeline...")
                self.pipeline = RAGPipeline(
                    retrieval_top_k=self.retrieval_top_k,
                    rerank_top_k=self.rerank_top_k,
                    model_key=self.model_key,
                    temperature=self.temperature,
                )
                logger.info("✅ RAG pipeline initialized successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
                raise
        return self.pipeline

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> dict:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        try:
            logger.info(f"🔍 Processing query: {question!r}")
            pipeline = self.get_pipeline()
            result = pipeline.ask(question, top_k=top_k, prompt_name=prompt_name)
            logger.info("✅ Query processed successfully.")
            return result
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
