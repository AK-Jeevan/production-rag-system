"""
RAGAS Evaluation Module
========================
Evaluates the RAG pipeline using RAGAS (RAG Assessment) metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

Uses langchain-compatible scorers to compute each metric per sample
and aggregate them into a summary report.
"""

import json
import csv
import os
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

from src.pipelines.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
Sample = Dict[str, str]          # {"question": ..., "reference": ...}
QueryFn = Callable[[str], Dict]  # question → {"answer": ..., "retrieved_docs": [...], "sources": [...]}
ScorerDict = Dict[str, Any]      # metric_name → scorer object with .score() method


class FakeScorer:
    """Fake scorer that always returns a fixed value — used when ragas is not installed."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def score(self, **kwargs) -> "SimpleNamespace":
        from types import SimpleNamespace
        # Validate expected keys are present
        assert kwargs.get("user_input"), "Missing 'user_input'"
        assert kwargs.get("response"), "Missing 'response'"
        assert kwargs.get("reference"), "Missing 'reference'"
        assert kwargs.get("retrieved_contexts"), "Missing 'retrieved_contexts'"
        return SimpleNamespace(value=self.value)


def _get_scorers() -> ScorerDict:
    """
    Try to import real RAGAS scorers.  If `ragas` is not installed, fall back
    to FakeScorer instances that return 0.0 so the evaluation pipeline still
    runs without crashing.
    """
    try:
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        logger.info("✅ Using real RAGAS scorers.")
        return {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
    except ImportError:
        logger.warning(
            "⚠️  ragas package not installed. Using fake scorers (scores = 0.0). "
            "Install with: pip install ragas"
        )
        return {
            "faithfulness": FakeScorer(0.0),
            "answer_relevancy": FakeScorer(0.0),
            "context_precision": FakeScorer(0.0),
            "context_recall": FakeScorer(0.0),
        }


class RagasEvaluator:
    """
    End-to-end RAGAS evaluator for the production RAG pipeline.

    Usage::

        evaluator = RagasEvaluator.from_project()
        samples = RagasEvaluator.load_dataset("evaluation/datasets/rag_eval.json")
        report = evaluator.evaluate(samples)
        json_path, csv_path = evaluator.save(report, "evaluation/results")
    """

    def __init__(
        self,
        query_fn: QueryFn,
        scorers: Optional[ScorerDict] = None,
    ):
        """
        Parameters
        ----------
        query_fn :
            A callable that accepts a question string and returns a dict with
            at least ``answer``, ``retrieved_docs`` (list of ``Document`` or
            objects with ``page_content``), and ``sources`` (list of str).
        scorers :
            A dict mapping metric names to scorer objects.  If ``None``, the
            default RAGAS scorers (or fake fallbacks) are used.
        """
        self.query_fn = query_fn
        self.scorers = scorers if scorers is not None else _get_scorers()

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_project(
        cls,
        model_name: str = "gemini-2.0-flash",
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> "RagasEvaluator":
        """
        Build an evaluator from the project's own ``RAGPipeline``.

        The pipeline is wrapped so that every call runs a full RAG query and
        returns the pieces the RAGAS scorers need.
        """
        pipeline = RAGPipeline(
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            model_key=_model_key_from_name(model_name),
        )

        def query_fn(question: str) -> dict:
            result = pipeline.ask(question)
            return {
                "answer": result["answer"],
                "retrieved_docs": result.get("retrieved_docs", []),
                "sources": result.get("sources", []),
            }

        logger.info(
            f"🔧 RAGAS evaluator created (model={model_name}, "
            f"retrieval_top_k={retrieval_top_k}, rerank_top_k={rerank_top_k})"
        )
        return cls(query_fn=query_fn)

    # ── Dataset I/O ────────────────────────────────────────────────────────────

    @staticmethod
    def load_dataset(path: str) -> List[Sample]:
        """
        Load evaluation samples from a JSON file.

        Expected format (list of dicts)::

            [
                {"question": "What is …?", "reference": "The expected answer …"},
                …
            ]

        Returns a list of samples; each sample must have ``question`` and
        ``reference`` keys.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"❌ Dataset not found at: {path_obj.resolve()}"
            )

        with path_obj.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            raise ValueError("❌ Dataset must be a JSON array of objects.")

        for i, sample in enumerate(samples):
            if "question" not in sample:
                raise ValueError(f"❌ Sample {i} is missing the 'question' field.")
            if "reference" not in sample:
                raise ValueError(
                    f"❌ Sample {i} is missing the 'reference' field. "
                    "RAGAS needs a ground-truth reference answer."
                )

        logger.info(f"📂 Loaded {len(samples)} evaluation samples from: {path}")
        return samples

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, samples: List[Sample]) -> Dict[str, Any]:
        """
        Run the RAGAS evaluation on every sample.

        For each sample:
          1. Query the RAG pipeline.
          2. Extract the generated answer and retrieved contexts.
          3. Score every metric using the configured scorers.

        Returns a report dict with ``sample_count``, ``summary`` (mean scores),
        and ``samples`` (per-sample scores).
        """
        logger.info(f"🚀 Starting RAGAS evaluation on {len(samples)} samples…")

        sample_scores: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples):
            question = sample["question"]
            reference = sample["reference"]

            logger.info(f"  [{idx + 1}/{len(samples)}] {question!r}")

            # --- Query the pipeline ---
            try:
                result = self.query_fn(question)
            except Exception as exc:
                logger.error(f"    ❌ Query failed: {exc}")
                # Record a failed sample with None scores
                sample_scores.append({
                    "question": question,
                    "reference": reference,
                    "response": "",
                    "contexts": [],
                    "sources": [],
                    **{name: None for name in self.scorers},
                })
                continue

            answer = result.get("answer", "")
            retrieved_docs: List = result.get("retrieved_docs", [])
            sources: List[str] = result.get("sources", [])

            # Extract page_content for contexts
            contexts = [
                doc.page_content if isinstance(doc, Document)
                else getattr(doc, "page_content", str(doc))
                for doc in retrieved_docs
            ]

            # --- Score each metric ---
            scores: Dict[str, Optional[float]] = {}
            for metric_name, scorer in self.scorers.items():
                try:
                    score_result = scorer.score(
                        user_input=question,
                        response=answer,
                        reference=reference,
                        retrieved_contexts=contexts,
                    )
                    scores[metric_name] = round(float(score_result.value), 4)
                except Exception as exc:
                    logger.warning(
                        f"    ⚠️  {metric_name} scoring failed: {exc}"
                    )
                    scores[metric_name] = None

            sample_scores.append({
                "question": question,
                "reference": reference,
                "response": answer,
                "contexts": contexts,
                "sources": sources,
                **scores,
            })

        # --- Aggregate summary ---
        summary: Dict[str, float] = {}
        for metric_name in self.scorers:
            valid = [
                s[metric_name]
                for s in sample_scores
                if s[metric_name] is not None
            ]
            summary[metric_name] = round(sum(valid) / len(valid), 4) if valid else 0.0

        report = {
            "sample_count": len(samples),
            "summary": summary,
            "samples": sample_scores,
        }

        logger.info("✅ RAGAS evaluation complete.")
        for metric, score in summary.items():
            logger.info(f"   {metric}: {score:.4f}")

        return report

    # ── Report I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def save(report: Dict[str, Any], output_dir: str) -> tuple:
        """
        Write the evaluation report as JSON and CSV.

        Parameters
        ----------
        report :
            The dict returned by ``evaluate()``.
        output_dir :
            Directory where files will be saved.

        Returns
        -------
        (json_path, csv_path) — both ``Path`` objects.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── JSON ───────────────────────────────────────────────────────────────
        json_path = out / "ragas_report.json"
        # Convert any Path / non-serialisable items
        serialisable = json.loads(json.dumps(report, default=str))
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 JSON report saved to: {json_path}")

        # ── CSV ────────────────────────────────────────────────────────────────
        csv_path = out / "ragas_report.csv"
        samples = report.get("samples", [])
        if samples:
            fieldnames = list(samples[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(samples)
            logger.info(f"📄 CSV report saved to:  {csv_path}")
        else:
            csv_path.write_text("")
            logger.warning("⚠️  No samples to write to CSV.")

        return json_path, csv_path


# ── Helper ────────────────────────────────────────────────────────────────────

def _model_key_from_name(model_name: str) -> str:
    """Map a Gemini model name back to the short key used by ``GeminiGenerator``."""
    mapping = {
        "gemini-2.0-flash": "flash",
        "gemini-2.5-flash": "flash25",
        "gemini-1.5-pro": "pro",
    }
    return mapping.get(model_name, "flash25")