import pytest
from unittest.mock import MagicMock, patch


class TestVectorStoreManager:
    @patch("src.vectorstore.vector_store.FAISS")
    def test_create_vector_store(self, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        embedding = MagicMock()
        mock_index = MagicMock()
        mock_faiss.from_documents.return_value = mock_index

        manager = VectorStoreManager(embedding_model=embedding)
        docs = [MagicMock(), MagicMock(), MagicMock()]

        result = manager.create_vector_store(docs)

        assert result is mock_index
        mock_faiss.from_documents.assert_called_once_with(
            documents=docs, embedding=embedding
        )

    @patch("src.vectorstore.vector_store.FAISS")
    def test_create_empty_docs_raises(self, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        manager = VectorStoreManager(embedding_model=MagicMock())
        with pytest.raises(ValueError):
            manager.create_vector_store([])

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.os.makedirs")
    def test_save_creates_index(self, mock_makedirs, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        manager = VectorStoreManager(embedding_model=MagicMock())
        mock_index = MagicMock()

        manager.save_vector_store(mock_index)

        mock_index.save_local.assert_called_once_with("models/faiss_index")
        mock_makedirs.assert_called_once_with("models/faiss_index", exist_ok=True)

    @patch("src.vectorstore.vector_store.FAISS")
    @patch("src.vectorstore.vector_store.os.path.exists", return_value=True)
    def test_load_existing_index(self, mock_exists, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        embedding = MagicMock()
        mock_faiss.load_local.return_value = MagicMock()
        manager = VectorStoreManager(embedding_model=embedding)

        manager.load_vector_store()

        mock_faiss.load_local.assert_called_once_with(
            "models/faiss_index",
            embedding,
            allow_dangerous_deserialization=True,
        )

    @patch("src.vectorstore.vector_store.FAISS")
    def test_similarity_search_returns_docs(self, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        mock_index = MagicMock()
        mock_index.similarity_search.return_value = [MagicMock(), MagicMock()]

        manager = VectorStoreManager(embedding_model=MagicMock())
        results = manager.similarity_search(mock_index, "What is FastAPI?", top_k=2)

        assert len(results) == 2
        mock_index.similarity_search.assert_called_once_with("What is FastAPI?", k=2)

    @patch("src.vectorstore.vector_store.FAISS")
    def test_similarity_search_empty_query_raises(self, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        manager = VectorStoreManager(embedding_model=MagicMock())
        with pytest.raises((ValueError, Exception)):
            manager.similarity_search(MagicMock(), "", top_k=5)

    @patch("src.vectorstore.vector_store.FAISS")
    def test_load_missing_index_raises(self, mock_faiss):
        from src.vectorstore.vector_store import VectorStoreManager

        with patch("src.vectorstore.vector_store.os.path.exists", return_value=False):
            manager = VectorStoreManager(embedding_model=MagicMock())
            with pytest.raises(FileNotFoundError):
                manager.load_vector_store()
