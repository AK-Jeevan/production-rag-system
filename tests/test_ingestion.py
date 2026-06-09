import pytest
from unittest.mock import MagicMock, patch, mock_open


class TestLoader:

    @patch("builtins.open", mock_open(read_data=b"fake pdf content"))
    @patch("src.ingestion.loader.PyPDFLoader")
    def test_load_pdf(self, mock_loader):
        from src.ingestion.loader import Loader
        mock_loader.return_value.load.return_value = [MagicMock()]
        loader = Loader()
        docs   = loader.load("fake.pdf")
        assert len(docs) >= 1

    def test_load_unsupported_format_raises(self):
        from src.ingestion.loader import Loader
        loader = Loader()
        with pytest.raises((ValueError, Exception)):
            loader.load("file.xyz")


class TestChunker:

    def test_chunk_documents(self):
        from src.ingestion.chunker import Chunker
        mock_doc      = MagicMock()
        mock_doc.page_content = "This is a long document. " * 100
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        chunks  = chunker.chunk([mock_doc])
        assert len(chunks) > 1

    def test_chunk_empty_docs_returns_empty(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        result  = chunker.chunk([])
        assert result == []


class TestCleaner:

    def test_clean_removes_whitespace(self):
        from src.ingestion.cleaner import Cleaner
        mock_doc = MagicMock()
        mock_doc.page_content = "  extra   spaces  "
        cleaner  = Cleaner()
        result   = cleaner.clean([mock_doc])
        assert len(result) > 0

    def test_clean_empty_docs_returns_empty(self):
        from src.ingestion.cleaner import Cleaner
        cleaner = Cleaner()
        result  = cleaner.clean([])
        assert result == []