import pytest
from prometheus_client import REGISTRY


class TestMetricsRegistered:
    """Verify all custom metrics are registered in the Prometheus registry."""

    def test_request_count_registered(self):
        from src.monitoring.metrics import REQUEST_COUNT
        assert REQUEST_COUNT is not None
        assert "rag_requests_total" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_error_count_registered(self):
        from src.monitoring.metrics import ERROR_COUNT
        assert ERROR_COUNT is not None
        assert "rag_errors_total" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_query_latency_registered(self):
        from src.monitoring.metrics import QUERY_LATENCY
        assert QUERY_LATENCY is not None
        assert "rag_query_latency_seconds" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_retrieval_latency_registered(self):
        from src.monitoring.metrics import RETRIEVAL_LATENCY
        assert RETRIEVAL_LATENCY is not None
        assert "retrieval_latency_seconds" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_rerank_latency_registered(self):
        from src.monitoring.metrics import RERANK_LATENCY
        assert RERANK_LATENCY is not None
        assert "rerank_latency_seconds" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_generation_latency_registered(self):
        from src.monitoring.metrics import GENERATION_LATENCY
        assert GENERATION_LATENCY is not None
        assert "generation_latency_seconds" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_input_tokens_registered(self):
        from src.monitoring.metrics import INPUT_TOKENS
        assert INPUT_TOKENS is not None
        assert "input_tokens_total" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_output_tokens_registered(self):
        from src.monitoring.metrics import OUTPUT_TOKENS
        assert OUTPUT_TOKENS is not None
        assert "output_tokens_total" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_estimated_cost_registered(self):
        from src.monitoring.metrics import ESTIMATED_COST
        assert ESTIMATED_COST is not None
        assert "estimated_cost_usd_total" in [
            m.name for m in REGISTRY.collect()
        ]

    def test_pipeline_info_registered(self):
        from src.monitoring.metrics import PIPELINE_INFO
        assert PIPELINE_INFO is not None
        assert "rag_pipeline_info" in [
            m.name for m in REGISTRY.collect()
        ]


class TestMetricsBehavior:
    """Verify metrics increment and observe correctly."""

    def test_request_count_increments(self):
        from src.monitoring.metrics import REQUEST_COUNT
        before = REQUEST_COUNT.labels(endpoint="/query")._value.get()
        REQUEST_COUNT.labels(endpoint="/query").inc()
        after  = REQUEST_COUNT.labels(endpoint="/query")._value.get()
        assert after == before + 1

    def test_error_count_increments(self):
        from src.monitoring.metrics import ERROR_COUNT
        before = ERROR_COUNT.labels(endpoint="/query", error_type="Exception")._value.get()
        ERROR_COUNT.labels(endpoint="/query", error_type="Exception").inc()
        after  = ERROR_COUNT.labels(endpoint="/query", error_type="Exception")._value.get()
        assert after == before + 1

    def test_input_tokens_increments(self):
        from src.monitoring.metrics import INPUT_TOKENS
        before = INPUT_TOKENS.labels(model="flash25")._value.get()
        INPUT_TOKENS.labels(model="flash25").inc(100)
        after  = INPUT_TOKENS.labels(model="flash25")._value.get()
        assert after == before + 100

    def test_output_tokens_increments(self):
        from src.monitoring.metrics import OUTPUT_TOKENS
        before = OUTPUT_TOKENS.labels(model="flash25")._value.get()
        OUTPUT_TOKENS.labels(model="flash25").inc(50)
        after  = OUTPUT_TOKENS.labels(model="flash25")._value.get()
        assert after == before + 50

    def test_estimated_cost_increments(self):
        from src.monitoring.metrics import ESTIMATED_COST
        before = ESTIMATED_COST.labels(model="flash25")._value.get()
        ESTIMATED_COST.labels(model="flash25").inc(0.001)
        after  = ESTIMATED_COST.labels(model="flash25")._value.get()
        assert round(after - before, 6) == 0.001

    def test_query_latency_observes(self):
        from src.monitoring.metrics import QUERY_LATENCY
        # observe() should not raise
        QUERY_LATENCY.labels(endpoint="/query").observe(1.5)

    def test_retrieval_latency_observes(self):
        from src.monitoring.metrics import RETRIEVAL_LATENCY
        RETRIEVAL_LATENCY.observe(0.25)

    def test_generation_latency_observes(self):
        from src.monitoring.metrics import GENERATION_LATENCY
        GENERATION_LATENCY.observe(2.5)

    def test_rerank_latency_observes(self):
        from src.monitoring.metrics import RERANK_LATENCY
        RERANK_LATENCY.observe(0.1)