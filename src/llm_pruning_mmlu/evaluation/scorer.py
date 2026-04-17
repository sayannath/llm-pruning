from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_pruning_mmlu.data.prompting import CHOICES, format_mmlu_prompt


def choice_token_ids(tokenizer, choices: list[str] | None = None) -> dict[str, list[int]]:
    choices = choices or CHOICES
    ids: dict[str, list[int]] = {}
    for choice in choices:
        encoded = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(choice, add_special_tokens=False)["input_ids"]
        ids[choice] = encoded
    return ids


@torch.no_grad()
def score_choice(model, tokenizer, prompt: str, choice: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    choice_ids = tokenizer(" " + choice, return_tensors="pt", add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    input_ids = torch.cat([prompt_ids, choice_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_len - 1, 0)
    return float(token_log_probs[:, start:].sum().item())


def predict_example(model, tokenizer, example: dict, choices: list[str] | None = None) -> dict:
    choices = choices or CHOICES
    prompt = format_mmlu_prompt(example["question"], example["choices"])
    scores = {choice: score_choice(model, tokenizer, prompt, choice) for choice in choices}
    pred = max(scores.items(), key=lambda item: item[1])[0]
    gold = example["answer"]
    return {
        "subject": example["subject"],
        "question_id": example["question_id"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold),
        "scores": scores,
    }
