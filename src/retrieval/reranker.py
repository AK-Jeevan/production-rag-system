import logging
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    AVAILABLE_MODELS = {
        "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # fast, lightweight
        "electra": "cross-encoder/ms-marco-electra-base",  # better quality
    }

    def __init__(self, model_key: str = "minilm", top_k: int = 5):
        self.top_k = top_k
        self.model_name = self.AVAILABLE_MODELS.get(model_key, model_key)

        logger.info(f"🔄 Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info(f"✅ Reranker loaded: {self.model_name}")

    def rerank(self, query: str, docs: list, top_k: int = None) -> list:
        """Rerank retrieved docs by relevance to query and return top_k."""
        if not docs:
            logger.warning("⚠️  No docs to rerank.")
            return []

        k = top_k or self.top_k

        # Score each doc against the query
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, score in scored_docs[:k]]
        top_scores = [round(float(score), 4) for _, score in scored_docs[:k]]

        logger.info(f"✅ Reranked {len(docs)} → top {k} docs.")
        logger.info(f"   Top scores: {top_scores}")

        return top_docs
