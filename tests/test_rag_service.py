import pytest
from unittest.mock import MagicMock, patch


class TestRAGService:
    @patch("app.services.rag_service.RAGPipeline")
    def test_init_success(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_pipeline.return_value = MagicMock()
        service = RAGService()
        assert service.pipeline is not None

    @patch("app.services.rag_service.RAGPipeline")
    def test_init_failure_raises(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_pipeline.side_effect = Exception("❌ Pipeline init failed")
        with pytest.raises(Exception, match="Pipeline init failed"):
            RAGService()

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_success(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_instance = MagicMock()
        mock_instance.ask.return_value = {
            "answer": "FastAPI is a web framework.",
            "sources": ["doc1.pdf"],
            "confidence": 0.95,
        }
        mock_pipeline.return_value = mock_instance

        service = RAGService()
        result = service.query("What is FastAPI?")

        assert result["answer"] == "FastAPI is a web framework."
        assert result["sources"] == ["doc1.pdf"]
        mock_instance.ask.assert_called_once_with(
            "What is FastAPI?", top_k=None, prompt_name=None
        )

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_passes_top_k(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_instance = MagicMock()
        mock_instance.ask.return_value = {"answer": "A", "sources": []}
        mock_pipeline.return_value = mock_instance

        service = RAGService()
        service.query("What is FastAPI?", top_k=10)
        mock_instance.ask.assert_called_once_with(
            "What is FastAPI?", top_k=10, prompt_name=None
        )

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_empty_question_raises(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_pipeline.return_value = MagicMock()
        service = RAGService()
        with pytest.raises(ValueError, match="Question must not be empty"):
            service.query("")

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_whitespace_question_raises(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_pipeline.return_value = MagicMock()
        service = RAGService()
        with pytest.raises(ValueError):
            service.query("   ")

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_pipeline_failure_raises(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_instance = MagicMock()
        mock_instance.ask.side_effect = Exception("LLM unavailable")
        mock_pipeline.return_value = mock_instance

        service = RAGService()
        with pytest.raises(Exception, match="LLM unavailable"):
            service.query("What is FastAPI?")

    @patch("app.services.rag_service.RAGPipeline")
    def test_query_returns_dict(self, mock_pipeline):
        from app.services.rag_service import RAGService

        mock_instance = MagicMock()
        mock_instance.ask.return_value = {"answer": "Answer", "sources": []}
        mock_pipeline.return_value = mock_instance

        service = RAGService()
        result = service.query("Some question")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
