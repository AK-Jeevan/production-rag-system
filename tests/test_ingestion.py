import pytest
from unittest.mock import MagicMock, patch, mock_open


class TestLoader:
    @patch("src.ingestion.loader.os.walk")
    @patch("src.ingestion.loader.os.path.exists", return_value=True)
    @patch("src.ingestion.loader.PyPDFLoader")
    def test_load_pdf(self, mock_loader, mock_exists, mock_walk):
        from src.ingestion.loader import DocumentLoader

        mock_walk.return_value = [("data", [], ["fake.pdf"])]
        mock_loader.return_value.load.return_value = [MagicMock()]

        with patch.dict(
            DocumentLoader.SUPPORTED_EXTENSIONS, {".pdf": mock_loader}, clear=False
        ):
            loader = DocumentLoader(data_path="data")
            docs = loader.load_documents()

        assert len(docs) >= 1
        mock_loader.assert_called_once_with("data\\fake.pdf")

    @patch("src.ingestion.loader.os.walk")
    @patch("src.ingestion.loader.os.path.exists", return_value=True)
    def test_load_unsupported_format_returns_empty(self, mock_exists, mock_walk):
        from src.ingestion.loader import DocumentLoader

        mock_walk.return_value = [("data", [], ["file.xyz"])]

        loader = DocumentLoader(data_path="data")
        docs = loader.load_documents()

        assert docs == []


class TestChunker:
    def test_chunk_documents(self):
        from src.ingestion.chunker import DocumentChunker
        from langchain_core.documents import Document

        mock_doc = Document(
            page_content="This is a long document. " * 100,
            metadata={"source": "test.txt"},
        )
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents([mock_doc])
        assert len(chunks) > 1

    def test_chunk_empty_docs_returns_empty(self):
        from src.ingestion.chunker import DocumentChunker

        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        result = chunker.split_documents([])
        assert result == []


class TestCleaner:
    def test_clean_removes_whitespace(self):
        from src.ingestion.cleaner import TextCleaner

        mock_doc = MagicMock()
        mock_doc.page_content = "  extra   spaces and enough text to keep  "
        cleaner = TextCleaner()
        result = cleaner.clean_documents([mock_doc])
        assert len(result) > 0
        assert result[0].page_content == "extra spaces and enough text to keep"

    def test_clean_empty_docs_returns_empty(self):
        from src.ingestion.cleaner import TextCleaner

        cleaner = TextCleaner()
        result = cleaner.clean_documents([])
        assert result == []
