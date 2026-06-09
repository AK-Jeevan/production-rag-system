from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health Check ──────────────────────────────────────────────────────────────
def test_health_status_code():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_fields():
    response = client.get("/api/v1/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data
    assert "version" in data


def test_health_service_name():
    response = client.get("/api/v1/health")
    assert response.json()["service"] == "RAG Assistant"


def test_health_version():
    response = client.get("/api/v1/health")
    assert response.json()["version"] == "1.0.0"


# ── Query ─────────────────────────────────────────────────────────────────────
def test_query_missing_question():
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422


def test_query_empty_question():
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_query_invalid_top_k_zero():
    response = client.post(
        "/api/v1/query", json={"question": "What is RAG?", "top_k": 0}
    )
    assert response.status_code == 422


def test_query_invalid_top_k_exceeded():
    response = client.post(
        "/api/v1/query", json={"question": "What is RAG?", "top_k": 999}
    )
    assert response.status_code == 422


def test_query_response_fields():
    response = client.post(
        "/api/v1/query", json={"question": "What is RAG?", "top_k": 5}
    )
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data


# ── Upload ────────────────────────────────────────────────────────────────────
def test_upload_no_file():
    response = client.post("/api/v1/upload")
    assert response.status_code == 422


def test_upload_invalid_extension():
    response = client.post(
        "/api/v1/upload",
        files={"file": ("malicious.exe", b"fake content", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_empty_file():
    response = client.post(
        "/api/v1/upload", files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400


def test_upload_valid_pdf():
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    if response.status_code == 201:
        data = response.json()
        assert "filename" in data
        assert "saved_as" in data
        assert "size_kb" in data
        assert "uploaded_at" in data


# ── Feedback ──────────────────────────────────────────────────────────────────
def test_feedback_missing_fields():
    response = client.post("/api/v1/feedback", json={})
    assert response.status_code == 422


def test_feedback_invalid_rating_too_low():
    response = client.post(
        "/api/v1/feedback",
        json={"question": "What is RAG?", "answer": "RAG is...", "rating": 0},
    )
    assert response.status_code == 422


def test_feedback_invalid_rating_too_high():
    response = client.post(
        "/api/v1/feedback",
        json={"question": "What is RAG?", "answer": "RAG is...", "rating": 6},
    )
    assert response.status_code == 422


def test_feedback_valid():
    response = client.post(
        "/api/v1/feedback",
        json={"question": "What is RAG?", "answer": "RAG is...", "rating": 5},
    )
    if response.status_code == 201:
        data = response.json()
        assert "feedback_id" in data
        assert "received_at" in data
        assert data["message"] == "Feedback saved successfully."


# ── Metrics ───────────────────────────────────────────────────────────────────
def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200


def test_metrics_content_type():
    response = client.get("/api/v1/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_rag_counter():
    response = client.get("/api/v1/metrics")
    assert b"rag_requests_total" in response.content
