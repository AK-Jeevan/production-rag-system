import os
import socket
import logging
import mlflow
import time

logger = logging.getLogger(__name__)


class _NullContext:
    """A no-op context manager returned when MLflow is disabled."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


class MLflowTracker:
    def __init__(self):
        self.enabled = False
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

        if not tracking_uri:
            logger.info("ℹ️ MLflow tracking disabled (MLFLOW_TRACKING_URI not set).")
            self.enabled = False
            self.run = None
            return

        # Quick connectivity test before committing to MLflow
        try:
            host, port = self._parse_uri(tracking_uri)
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("production-rag-system")
            logger.info(f"✅ MLflow connected at {tracking_uri}")
            self.enabled = True
        except Exception as e:
            logger.warning(
                f"⚠️ MLflow not available at {tracking_uri} ({e}). "
                f"Running without MLflow tracking."
            )
            self.enabled = False
        self.run = None

    @staticmethod
    def _parse_uri(uri: str):
        """Extract host and port from a tracking URI like http://host:port."""
        from urllib.parse import urlparse
        parsed = urlparse(uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5000
        return host, port

    def start_run(self, run_name: str):
        if not self.enabled:
            return _NullContext()
        try:
            self.run = mlflow.start_run(run_name=run_name)
            return self.run
        except Exception as e:
            logger.warning(f"⚠️ MLflow start_run failed: {e}. Disabling tracking.")
            self.enabled = False
            self.run = None
            return _NullContext()

    def log_rag_parameters(
        self,
        chunk_size,
        chunk_overlap,
        embedding_model,
        top_k,
        vector_db,
        llm_model,
        prompt_name=None,
    ):
        if not self.enabled:
            return
        try:
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)
            mlflow.log_param("embedding_model", embedding_model)
            mlflow.log_param("top_k", top_k)
            mlflow.log_param("vector_db", vector_db)
            mlflow.log_param("llm_model", llm_model)
            if prompt_name is not None:
                mlflow.log_param("prompt_name", prompt_name)
        except Exception as e:
            logger.warning(f"⚠️ MLflow log params failed: {e}")

    def log_latency_metrics(self, retrieval_latency, generation_latency, total_latency):
        if not self.enabled:
            return
        try:
            mlflow.log_metric("retrieval_latency", retrieval_latency)
            mlflow.log_metric("generation_latency", generation_latency)
            mlflow.log_metric("total_latency", total_latency)
        except Exception as e:
            logger.warning(f"⚠️ MLflow log latency failed: {e}")

    def log_token_metrics(self, input_tokens, output_tokens, total_tokens):
        if not self.enabled:
            return
        try:
            mlflow.log_metric("input_tokens", input_tokens)
            mlflow.log_metric("output_tokens", output_tokens)
            mlflow.log_metric("total_tokens", total_tokens)
        except Exception as e:
            logger.warning(f"⚠️ MLflow log tokens failed: {e}")

    def log_evaluation_metrics(self, answer_relevance, faithfulness):
        if not self.enabled:
            return
        try:
            mlflow.log_metric("answer_relevance", answer_relevance)
            mlflow.log_metric("faithfulness", faithfulness)
        except Exception as e:
            logger.warning(f"⚠️ MLflow log eval metrics failed: {e}")

    def log_metric(self, key: str, value: float) -> None:
        """Log a single metric to MLflow."""
        if not self.enabled:
            return
        try:
            mlflow.log_metric(key, value)
        except Exception as e:
            logger.warning(f"⚠️ MLflow log metric '{key}' failed: {e}")

    def end_run(self):
        if not self.enabled:
            return
        try:
            mlflow.end_run()
        except Exception:
            pass
        self.run = None


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = MLflowTracker()

    with tracker.start_run("test-run"):
        tracker.log_rag_parameters(
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="all-MiniLM-L6-v2",
            top_k=5,
            vector_db="chromadb",
            llm_model="gpt-4",
        )

        # Simulate retrieval latency
        retrieval_start = time.time()
        time.sleep(0.3)
        retrieval_latency = time.time() - retrieval_start

        # Simulate generation latency
        generation_start = time.time()
        time.sleep(1.2)
        generation_latency = time.time() - generation_start

        total_latency = retrieval_latency + generation_latency

        tracker.log_latency_metrics(
            retrieval_latency=round(retrieval_latency, 4),
            generation_latency=round(generation_latency, 4),
            total_latency=round(total_latency, 4),
        )

        tracker.log_token_metrics(input_tokens=120, output_tokens=80, total_tokens=200)

        tracker.log_evaluation_metrics(answer_relevance=0.87, faithfulness=0.92)

        # Test individual metric logging
        tracker.log_metric("docs_retrieved", 20)
        tracker.log_metric("docs_reranked", 5)

        print("✅ Run logged successfully.")
        print(f"   Retrieval latency : {retrieval_latency:.4f}s")
        print(f"   Generation latency: {generation_latency:.4f}s")
        print(f"   Total latency     : {total_latency:.4f}s")
        print("   Check http://127.0.0.1:5000 to see the run.")
