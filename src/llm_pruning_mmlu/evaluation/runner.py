from __future__ import annotations

import time

from tqdm import tqdm

from llm_pruning_mmlu.evaluation.metrics import compute_metrics
from llm_pruning_mmlu.evaluation.scorer import predict_example


def evaluate_examples(
    model, tokenizer, examples: list[dict], answer_choices: list[str]
) -> tuple[dict, list[dict]]:
    predictions = []
    for example in tqdm(examples, desc="Evaluating", leave=False):
        t0 = time.perf_counter()
        pred = predict_example(model, tokenizer, example, answer_choices)
        pred["elapsed_s"] = round(time.perf_counter() - t0, 6)
        predictions.append(pred)
    return compute_metrics(predictions), predictions
