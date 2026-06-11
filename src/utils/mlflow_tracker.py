import os
import mlflow
import time


class MLflowTracker:
    def __init__(self):
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        )
        mlflow.set_experiment("production-rag-system")
        self.run = None

    def start_run(self, run_name: str):
        self.run = mlflow.start_run(run_name=run_name)
        return self.run

    def log_rag_parameters(
        self, chunk_size, chunk_overlap, embedding_model, top_k, vector_db, llm_model
    ):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("vector_db", vector_db)
        mlflow.log_param("llm_model", llm_model)

    def log_latency_metrics(self, retrieval_latency, generation_latency, total_latency):
        mlflow.log_metric("retrieval_latency", retrieval_latency)
        mlflow.log_metric("generation_latency", generation_latency)
        mlflow.log_metric("total_latency", total_latency)

    def log_token_metrics(self, input_tokens, output_tokens, total_tokens):
        mlflow.log_metric("input_tokens", input_tokens)
        mlflow.log_metric("output_tokens", output_tokens)
        mlflow.log_metric("total_tokens", total_tokens)

    def log_evaluation_metrics(self, answer_relevance, faithfulness):
        mlflow.log_metric("answer_relevance", answer_relevance)
        mlflow.log_metric("faithfulness", faithfulness)

    def log_metric(self, key: str, value: float) -> None:
        """Log a single metric to MLflow."""
        mlflow.log_metric(key, value)  # ← fix: removed stray code below

    def end_run(self):
        mlflow.end_run()
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
