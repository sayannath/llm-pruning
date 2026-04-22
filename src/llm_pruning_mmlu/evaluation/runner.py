from __future__ import annotations

import time

from tqdm import tqdm

from llm_pruning_mmlu.evaluation.metrics import compute_metrics
from llm_pruning_mmlu.evaluation.scorer import predict_example, predict_example_generation


def evaluate_examples(
    model,
    tokenizer,
    examples: list[dict],
    answer_choices: list[str],
    scoring_mode: str = "choice_logprob",
) -> tuple[dict, list[dict]]:
    if scoring_mode == "generation":
        _predict = lambda ex: predict_example_generation(model, tokenizer, ex, answer_choices)
    else:
        _predict = lambda ex: predict_example(model, tokenizer, ex, answer_choices)

    predictions = []
    for example in tqdm(examples, desc="Evaluating", leave=False):
        t0 = time.perf_counter()
        pred = _predict(example)
        pred["elapsed_s"] = round(time.perf_counter() - t0, 6)
        predictions.append(pred)
    return compute_metrics(predictions), predictions
