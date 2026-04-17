from __future__ import annotations

from llm_pruning_mmlu.data.mmlu import load_mmlu
from llm_pruning_mmlu.evaluation.runner import evaluate_examples
from llm_pruning_mmlu.models.loader import DummyCausalLM, DummyTokenizer


def test_smoke_eval_tiny_model():
    examples = load_mmlu(fixture_path="tests/fixtures/tiny_mmlu.jsonl")
    metrics, predictions = evaluate_examples(
        DummyCausalLM(), DummyTokenizer(), examples, ["A", "B", "C", "D"]
    )
    assert metrics["num_samples"] == 2
    assert len(predictions) == 2
    assert set(predictions[0]["scores"]) == {"A", "B", "C", "D"}
