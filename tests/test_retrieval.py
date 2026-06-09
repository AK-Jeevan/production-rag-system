import pytest
from unittest.mock import MagicMock, patch


class TestRetriever:
    @patch("src.retrieval.retriever.HybridRetriever")
    def test_retrieve_returns_docs(self, mock_hybrid_retriever):
        from src.retrieval.retriever import Retriever

        mock_instance = MagicMock()
        mock_instance.retrieve.return_value = [MagicMock(), MagicMock()]
        mock_hybrid_retriever.return_value = mock_instance

        retriever = Retriever(top_k=5)
        docs = retriever.retrieve("What is FastAPI?")
        assert len(docs) == 2
        mock_instance.retrieve.assert_called_once_with("What is FastAPI?", top_k=5)

    @patch("src.retrieval.retriever.HybridRetriever")
    def test_retrieve_empty_query_raises(self, mock_hybrid_retriever):
        from src.retrieval.retriever import Retriever

        retriever = Retriever(top_k=5)
        with pytest.raises(ValueError):
            retriever.retrieve("")


class TestReranker:
    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_cross_encoder):
        from src.retrieval.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.3, 0.2]
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker(top_k=2)
        docs = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        result = reranker.rerank("query", docs, top_k=2)
        assert result == [docs[1], docs[2]]

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_docs_returns_empty(self, mock_cross_encoder):
        from src.retrieval.reranker import Reranker

        mock_cross_encoder.return_value = MagicMock()
        reranker = Reranker(top_k=5)
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


class TestQueryRewriter:
    @patch("src.retrieval.query_rewriter.genai.GenerativeModel")
    def test_rewrite_with_history(self, mock_model):
        from src.retrieval.query_rewriter import QueryRewriter

        mock_response = MagicMock()
        mock_response.text = "Rewritten query"
        mock_model.return_value.generate_content.return_value = mock_response

        rewriter = QueryRewriter()
        result = rewriter.rewrite(
            question="How does it work?", history="User asked about FastAPI."
        )
        assert result == "Rewritten query"

    @patch("src.retrieval.query_rewriter.genai.GenerativeModel")
    def test_rewrite_no_history_uses_default_history(self, mock_model):
        from src.retrieval.query_rewriter import QueryRewriter

        mock_response = MagicMock()
        mock_response.text = "Rewritten without history"
        mock_model.return_value.generate_content.return_value = mock_response

        rewriter = QueryRewriter()
        result = rewriter.rewrite(question="What is FastAPI?", history="")
        assert result == "Rewritten without history"
        prompt = mock_model.return_value.generate_content.call_args.args[0]
        assert "No prior conversation." in prompt
        assert "What is FastAPI?" in prompt
