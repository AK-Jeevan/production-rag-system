import argparse
import logging

from src.evaluation.ragas_evaluator import RagasEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/rag_eval.json",
        help="JSON dataset containing question and reference fields.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory for JSON and CSV evaluation reports.",
    )
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    samples = RagasEvaluator.load_dataset(args.dataset)
    evaluator = RagasEvaluator.from_project(
        model_name=args.model,
        retrieval_top_k=args.retrieval_top_k,
        rerank_top_k=args.rerank_top_k,
    )
    report = evaluator.evaluate(samples)
    json_path, csv_path = evaluator.save(report, args.output_dir)

    print("RAGAS evaluation complete")
    for metric, score in report["summary"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
