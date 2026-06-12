import pytest
from unittest.mock import MagicMock, patch


class TestGeminiGenerator:
    @patch("src.generation.generator._get_client")
    def test_init_success(self, mock_client):
        from src.generation.generator import GeminiGenerator

        gen = GeminiGenerator(model_key="flash25")
        assert gen.model_name == "gemini-2.5-flash"

    @patch("src.generation.generator._get_client")
    def test_invalid_model_key_falls_back(self, mock_client):
        from src.generation.generator import GeminiGenerator

        gen = GeminiGenerator(model_key="unknown-key")
        assert gen.model_name == "unknown-key"

    @patch("src.generation.generator._get_client")
    def test_generate_answer_success(self, mock_client):
        from src.generation.generator import GeminiGenerator

        mock_response = MagicMock()
        mock_response.text = "FastAPI is a web framework."
        mock_client.return_value.models.generate_content.return_value = mock_response

        gen = GeminiGenerator()
        answer = gen.generate_answer("What is FastAPI?")
        assert answer == "FastAPI is a web framework."

    @patch("src.generation.generator._get_client")
    def test_generate_answer_empty_prompt_raises(self, mock_client):
        from src.generation.generator import GeminiGenerator

        gen = GeminiGenerator()
        with pytest.raises(ValueError):
            gen.generate_answer("")

    @patch("src.generation.generator._get_client")
    def test_generate_answer_stream_yields_chunks(self, mock_client):
        from src.generation.generator import GeminiGenerator

        chunk1, chunk2 = MagicMock(), MagicMock()
        chunk1.text = "Hello "
        chunk2.text = "World"
        mock_client.return_value.models.generate_content_stream.return_value = [
            chunk1,
            chunk2,
        ]

        gen = GeminiGenerator()
        result = list(gen.generate_answer_stream("What is FastAPI?"))
        assert result == ["Hello ", "World"]

    @patch("src.generation.generator._get_client")
    def test_generate_rag_answer_empty_context_raises(self, mock_client):
        from src.generation.generator import GeminiGenerator

        gen = GeminiGenerator()
        with pytest.raises(ValueError):
            gen.generate_rag_answer(question="Q", context="")

    @patch("src.generation.generator._get_client")
    def test_list_available_models(self, mock_client):
        from src.generation.generator import GeminiGenerator

        models = GeminiGenerator.list_available_models()
        assert "flash25" in models
        assert "pro" in models
