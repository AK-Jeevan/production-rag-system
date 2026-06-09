import pytest
from unittest.mock import MagicMock, patch


class TestRetriever:

    @patch("src.retrieval.retriever.FAISS")
    def test_retrieve_returns_docs(self, mock_faiss):
        from src.retrieval.retriever import Retriever
        mock_instance = MagicMock()
        mock_instance.similarity_search.return_value = [MagicMock(), MagicMock()]
        mock_faiss.load_local.return_value = mock_instance

        retriever = Retriever(top_k=5)
        docs      = retriever.retrieve("What is FastAPI?")
        assert len(docs) == 2

    @patch("src.retrieval.retriever.FAISS")
    def test_retrieve_empty_query_raises(self, mock_faiss):
        from src.retrieval.retriever import Retriever
        retriever = Retriever(top_k=5)
        with pytest.raises(ValueError):
            retriever.retrieve("")


class TestReranker:

    def test_rerank_returns_top_k(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker(top_k=2)
        docs     = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        result   = reranker.rerank("query", docs, top_k=2)
        assert len(result) <= 2

    def test_rerank_empty_docs_returns_empty(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker(top_k=5)
        result   = reranker.rerank("query", [], top_k=5)
        assert result == []


class TestQueryRewriter:

    @patch("src.retrieval.query_rewriter.GeminiGenerator")
    def test_rewrite_with_history(self, mock_gen):
        from src.retrieval.query_rewriter import QueryRewriter
        mock_gen.return_value.generate_answer.return_value = "Rewritten query"
        rewriter = QueryRewriter()
        result   = rewriter.rewrite(question="How does it work?", history="User asked about FastAPI.")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.retrieval.query_rewriter.GeminiGenerator")
    def test_rewrite_no_history_returns_original(self, mock_gen):
        from src.retrieval.query_rewriter import QueryRewriter
        rewriter = QueryRewriter()
        result   = rewriter.rewrite(question="What is FastAPI?", history="")
        assert result == "What is FastAPI?"