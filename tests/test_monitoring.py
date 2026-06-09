import pytest
from prometheus_client import REGISTRY
from src.monitoring.metrics import (
    REQUEST_COUNT,
    ERROR_COUNT,
    DOCUMENT_UPLOAD_COUNT,
    QUERY_LATENCY,
    RETRIEVAL_LATENCY,
    RERANK_LATENCY,
    GENERATION_LATENCY,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    ESTIMATED_COST,
    PIPELINE_INFO,
)


def get_metric_names():
    """Return all registered metric base names."""
    return [m.name for m in REGISTRY.collect()]


class TestMetricsRegistered:

    def test_request_count_registered(self):
        # Counters are stored as base name without _total in REGISTRY
        assert "rag_requests" in get_metric_names()

    def test_error_count_registered(self):
        assert "rag_errors" in get_metric_names()

    def test_document_upload_count_registered(self):
        assert "documents_uploaded" in get_metric_names()

    def test_query_latency_registered(self):
        assert "rag_query_latency_seconds" in get_metric_names()

    def test_retrieval_latency_registered(self):
        assert "retrieval_latency_seconds" in get_metric_names()

    def test_rerank_latency_registered(self):
        assert "rerank_latency_seconds" in get_metric_names()

    def test_generation_latency_registered(self):
        assert "generation_latency_seconds" in get_metric_names()

    def test_input_tokens_registered(self):
        assert "input_tokens" in get_metric_names()

    def test_output_tokens_registered(self):
        assert "output_tokens" in get_metric_names()

    def test_estimated_cost_registered(self):
        assert "estimated_cost_usd" in get_metric_names()

    def test_pipeline_info_registered(self):
        assert "rag_pipeline" in get_metric_names()
