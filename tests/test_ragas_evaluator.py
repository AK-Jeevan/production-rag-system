import json
from types import SimpleNamespace

import pytest

from src.evaluation.ragas_evaluator import RagasEvaluator


class FakeScorer:
    def __init__(self, value):
        self.value = value

    def score(self, **kwargs):
        assert kwargs["user_input"]
        assert kwargs["response"]
        assert kwargs["reference"]
        assert kwargs["retrieved_contexts"]
        return SimpleNamespace(value=self.value)


def test_load_dataset_validates_required_fields(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([{"question": "Q", "reference": "A"}]))
    assert RagasEvaluator.load_dataset(path)[0]["question"] == "Q"

    path.write_text(json.dumps([{"question": "Q"}]))
    with pytest.raises(ValueError, match="reference"):
        RagasEvaluator.load_dataset(path)


def test_evaluate_returns_sample_and_summary_scores():
    document = SimpleNamespace(page_content="Grounded context")
    evaluator = RagasEvaluator(
        query_fn=lambda question: {
            "answer": "Grounded answer",
            "retrieved_docs": [document],
            "sources": ["guide.pdf"],
        },
        scorers={
            "faithfulness": FakeScorer(0.8),
            "answer_relevancy": FakeScorer(0.6),
        },
    )

    report = evaluator.evaluate([{"question": "Q", "reference": "A"}])

    assert report["sample_count"] == 1
    assert report["summary"] == {
        "faithfulness": 0.8,
        "answer_relevancy": 0.6,
    }
    assert report["samples"][0]["contexts"] == ["Grounded context"]


def test_save_writes_json_and_csv(tmp_path):
    report = {
        "summary": {"faithfulness": 1.0},
        "samples": [
            {
                "question": "Q",
                "reference": "A",
                "response": "A",
                "sources": ["doc.pdf"],
                "faithfulness": 1.0,
            }
        ],
    }
    json_path, csv_path = RagasEvaluator.save(report, tmp_path)
    assert json_path.exists()
    assert csv_path.exists()
    assert "faithfulness" in csv_path.read_text()
