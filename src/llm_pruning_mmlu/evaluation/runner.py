from __future__ import annotations

import time

from tqdm import tqdm

from llm_pruning_mmlu.evaluation.metrics import compute_metrics
from llm_pruning_mmlu.evaluation.scorer import predict_batch_generation, predict_example, predict_example_generation

_GENERATION_BATCH_SIZE = 4


def evaluate_examples(
    model,
    tokenizer,
    examples: list[dict],
    answer_choices: list[str],
    scoring_mode: str = "choice_logprob",
) -> tuple[dict, list[dict]]:
    if scoring_mode == "generation":
        predictions = []
        for i in tqdm(range(0, len(examples), _GENERATION_BATCH_SIZE), desc="Evaluating", leave=False):
            batch = examples[i:i + _GENERATION_BATCH_SIZE]
            t0 = time.perf_counter()
            batch_preds = predict_batch_generation(model, tokenizer, batch, answer_choices)
            elapsed = round((time.perf_counter() - t0) / len(batch), 6)
            for pred in batch_preds:
                pred["elapsed_s"] = elapsed
                predictions.append(pred)
        return compute_metrics(predictions), predictions

    if scoring_mode == "choice_logprob":
        _predict = lambda ex: predict_example(model, tokenizer, ex, answer_choices)
    else:
        _predict = lambda ex: predict_example_generation(model, tokenizer, ex, answer_choices)

    predictions = []
    for example in tqdm(examples, desc="Evaluating", leave=False):
        t0 = time.perf_counter()
        pred = _predict(example)
        pred["elapsed_s"] = round(time.perf_counter() - t0, 6)
        predictions.append(pred)
    return compute_metrics(predictions), predictions
