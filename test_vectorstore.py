import pytest
from unittest.mock import MagicMock, patch, mock_open


class TestVectorStore:

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_init_success(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        store = VectorStore()
        assert store is not None

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_build_from_docs(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        mock_faiss.from_documents.return_value = MagicMock()
        store = VectorStore()
        docs  = [MagicMock(), MagicMock(), MagicMock()]
        store.build(docs)
        mock_faiss.from_documents.assert_called_once()

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_build_empty_docs_raises(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        store = VectorStore()
        with pytest.raises((ValueError, Exception)):
            store.build([])

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_save_creates_index(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        mock_index = MagicMock()
        mock_faiss.from_documents.return_value = mock_index
        store      = VectorStore()
        docs       = [MagicMock()]
        store.build(docs)
        store.save("models/faiss_index")
        mock_index.save_local.assert_called_once_with("models/faiss_index")

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_load_existing_index(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        mock_faiss.load_local.return_value = MagicMock()
        store = VectorStore()
        store.load("models/faiss_index")
        mock_faiss.load_local.assert_called_once()

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_search_returns_docs(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = [MagicMock(), MagicMock()]
        mock_faiss.load_local.return_value = mock_index

        store = VectorStore()
        store.load("models/faiss_index")
        results = store.search("What is FastAPI?", top_k=2)
        assert len(results) == 2

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_search_empty_query_raises(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        mock_faiss.load_local.return_value = MagicMock()
        store = VectorStore()
        store.load("models/faiss_index")
        with pytest.raises((ValueError, Exception)):
            store.search("", top_k=5)

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.HuggingFaceEmbeddings")
    def test_search_before_load_raises(self, mock_embeddings, mock_faiss):
        from src.vectorstore.vector_store import VectorStore
        store = VectorStore()
        with pytest.raises((ValueError, AttributeError, Exception)):
            store.search("What is FastAPI?", top_k=5)