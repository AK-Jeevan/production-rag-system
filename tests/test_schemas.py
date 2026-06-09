import pytest
from pydantic import ValidationError
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.feedback import FeedbackRequest


# ── QueryRequest ──────────────────────────────────────────────────────────────
class TestQueryRequest:

    def test_valid_query_request(self):
        req = QueryRequest(question="What is FastAPI?")
        assert req.question == "What is FastAPI?"
        assert req.top_k == 5

    def test_custom_top_k(self):
        req = QueryRequest(question="What is FastAPI?", top_k=10)
        assert req.top_k == 10

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_top_k_below_min_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is FastAPI?", top_k=0)

    def test_top_k_above_max_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is FastAPI?", top_k=999)


# ── QueryResponse ─────────────────────────────────────────────────────────────
class TestQueryResponse:

    def test_valid_query_response(self):
        res = QueryResponse(answer="FastAPI is a web framework.", sources=[])
        assert res.answer == "FastAPI is a web framework."
        assert res.sources == []
        assert res.confidence is None

    def test_with_confidence(self):
        res = QueryResponse(answer="Answer", sources=["doc1.pdf"], confidence=0.95)
        assert res.confidence == 0.95

    def test_confidence_above_max_raises(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="Answer", sources=[], confidence=1.5)

    def test_confidence_below_min_raises(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="Answer", sources=[], confidence=-0.1)


# ── FeedbackRequest ───────────────────────────────────────────────────────────
class TestFeedbackRequest:

    def test_valid_feedback(self):
        req = FeedbackRequest(question="What is FastAPI?", answer="A framework.", rating=5)
        assert req.rating == 5

    def test_rating_below_min_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(question="Q", answer="A", rating=0)

    def test_rating_above_max_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(question="Q", answer="A", rating=6)

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(question="", answer="A", rating=3)