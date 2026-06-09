from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Docs & Schema Endpoints ───────────────────────────────────────────────────
def test_swagger_docs():
    """Swagger UI should be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc():
    """ReDoc UI should be accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema():
    """OpenAPI JSON schema should be accessible and valid."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    schema = response.json()
    assert schema["info"]["title"] == "Production RAG System"
    assert schema["info"]["version"] == "1.0.0"


# ── Root Endpoint ─────────────────────────────────────────────────────────────
def test_root():
    """Root endpoint should return HTML with correct content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Production RAG System" in response.text


# ── Health Endpoint ───────────────────────────────────────────────────────────
def test_health_check():
    """Health endpoint should return healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data
    assert "version" in data


# ── Metrics Endpoint ──────────────────────────────────────────────────────────
def test_metrics():
    """Prometheus metrics endpoint should return valid text/plain content."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert b"rag_requests_total" in response.content


# ── Query Endpoint Validation ─────────────────────────────────────────────────
def test_query_empty_question():
    """Empty question should return 422 Unprocessable Entity."""
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_question():
    """Missing question field should return 422 Unprocessable Entity."""
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422


def test_query_invalid_top_k():
    """top_k below 1 should return 422 Unprocessable Entity."""
    response = client.post(
        "/api/v1/query", json={"question": "What is FastAPI?", "top_k": 0}
    )
    assert response.status_code == 422


# ── Upload Endpoint Validation ────────────────────────────────────────────────
def test_upload_invalid_file_type():
    """Unsupported file type should return 415."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.exe", b"fake content", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_empty_file():
    """Empty file should return 400."""
    response = client.post(
        "/api/v1/upload", files={"file": ("test.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400


# ── Feedback Endpoint Validation ──────────────────────────────────────────────
def test_feedback_invalid_rating():
    """Rating outside 1-5 should return 422 Unprocessable Entity."""
    response = client.post(
        "/api/v1/feedback",
        json={"question": "What is RAG?", "answer": "RAG is...", "rating": 10},
    )
    assert response.status_code == 422


def test_feedback_missing_fields():
    """Missing required fields should return 422 Unprocessable Entity."""
    response = client.post("/api/v1/feedback", json={"question": "What is RAG?"})
    assert response.status_code == 422
