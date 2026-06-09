import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def mock_rag_result():
    return {
        "answer": "FastAPI is a web framework.",
        "sources": ["doc1.pdf"],
        "confidence": 0.95,
    }


# ── /query ────────────────────────────────────────────────────────────────────
class TestQueryRoute:
    @pytest.mark.asyncio
    async def test_query_success(self, mock_rag_result):
        mock_service = MagicMock()
        mock_service.query.return_value = mock_rag_result

        with patch("app.api.routes.get_rag_service", return_value=mock_service):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/query", json={"question": "What is FastAPI?"}
                )
        assert response.status_code == 200
        assert response.json()["answer"] == "FastAPI is a web framework."
        assert response.json()["sources"] == ["doc1.pdf"]

    @pytest.mark.asyncio
    async def test_query_empty_question_raises(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_service_error_returns_500(self):
        mock_service = MagicMock()
        mock_service.query.side_effect = Exception("LLM failed")

        with patch("app.api.routes.get_rag_service", return_value=mock_service):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/query", json={"question": "What is FastAPI?"}
                )
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_query_value_error_returns_422(self):
        mock_service = MagicMock()
        mock_service.query.side_effect = ValueError("Empty question")

        with patch("app.api.routes.get_rag_service", return_value=mock_service):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/query", json={"question": "What is FastAPI?"}
                )
        assert response.status_code == 422


# ── /health ───────────────────────────────────────────────────────────────────
class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_health_check(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
        assert "service" in response.json()


# ── /metrics ──────────────────────────────────────────────────────────────────
class TestMetricsRoute:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert "rag_requests_total" in response.text
