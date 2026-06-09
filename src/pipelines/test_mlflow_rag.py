import time
from src.utils.mlflow_tracker import MLflowTracker


def main():
    tracker = MLflowTracker()

    with tracker.start_run("rag_experiment_1"):
        # --- Parameters ---
        tracker.log_rag_parameters(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="all-MiniLM-L6-v2",
            top_k=5,
            vector_db="FAISS",
            llm_model="gpt-4",
        )

        # --- Retrieval latency ---
        retrieval_start = time.time()
        time.sleep(1.2)  # replace with real retrieval
        retrieval_latency = time.time() - retrieval_start

        # --- Generation latency ---
        generation_start = time.time()
        time.sleep(2.4)  # replace with real LLM call
        generation_latency = time.time() - generation_start

        total_latency = retrieval_latency + generation_latency

        tracker.log_latency_metrics(
            retrieval_latency=round(retrieval_latency, 4),
            generation_latency=round(generation_latency, 4),
            total_latency=round(total_latency, 4),
        )

        # --- Token usage ---
        tracker.log_token_metrics(
            input_tokens=850, output_tokens=420, total_tokens=1270
        )

        # --- Evaluation scores ---
        tracker.log_evaluation_metrics(answer_relevance=0.91, faithfulness=0.88)

        print("✅ Experiment logged successfully.")
        print(f"   Retrieval latency : {retrieval_latency:.4f}s")
        print(f"   Generation latency: {generation_latency:.4f}s")
        print(f"   Total latency     : {total_latency:.4f}s")
        print("   Total tokens      : 1270")
        print("   Answer relevance  : 0.91")
        print("   Faithfulness      : 0.88")
        print("   Check http://127.0.0.1:5000 to see the run.")
    # run auto-closes here — no need for tracker.end_run()


if __name__ == "__main__":
    main()
