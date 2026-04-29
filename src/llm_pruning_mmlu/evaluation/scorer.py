from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_pruning_mmlu.data.prompting import CHOICES, format_mmlu_prompt, normalize_answer


def choice_token_ids(tokenizer, choices: list[str] | None = None) -> dict[str, list[int]]:
    choices = choices or CHOICES
    ids: dict[str, list[int]] = {}
    for choice in choices:
        encoded = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
        if not encoded:
            encoded = tokenizer(choice, add_special_tokens=False)["input_ids"]
        ids[choice] = encoded
    return ids


def _prepare_prompt(tokenizer, raw_prompt: str) -> tuple[str, bool]:
    """Return (prompt_string, add_special_tokens) for log-prob scoring.

    Uses tokenize=False so the caller re-tokenizes the string — this path
    works correctly for Llama3.1 and Qwen3 whose tokenizers handle their
    own special tokens even with add_special_tokens=False.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, False
    return raw_prompt, True


def _tokenize_prompt_for_generation(tokenizer, raw_prompt: str) -> torch.Tensor:
    """Return prompt token IDs for generation-based scoring.

    Uses apply_chat_template with tokenize=True to correctly handle
    tokenizers (e.g. Gemma4) whose special tokens are silently dropped
    when the template string is re-tokenized with add_special_tokens=False.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        return ids
    return tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]


@torch.no_grad()
def score_choice(
    model, tokenizer, prompt: str, choice: str, add_special_tokens: bool = True
) -> float:
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=add_special_tokens
    )["input_ids"]
    choice_ids = tokenizer(
        " " + choice, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    device = next(model.parameters()).device
    input_ids = torch.cat([prompt_ids, choice_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_len - 1, 0)
    n_choice_tokens = choice_ids.shape[1]
    raw = float(token_log_probs[:, start:].sum().item())
    return raw / max(n_choice_tokens, 1)


def _parse_choice(text: str, choices: list[str]) -> str | None:
    """Extract the answer letter from free-form model output.

    Tries progressively looser patterns so that responses like
    "The correct answer is **C. Paris**." return 'C' rather than
    the first incidentally-matching letter in a word like "correct".
    """
    import re
    choice_pat = "[" + "".join(choices) + "]"
    # Try patterns from most to least specific
    for pat in [
        rf"(?:answer|option)\s*(?:is|:)\s*\*{{0,2}}({choice_pat})\b",  # "answer is C"
        rf"\*\*({choice_pat})[.\s*]",       # **C. or **C**
        rf"\b({choice_pat})\.",             # C. (letter-dot)
        rf"\(({choice_pat})\)",             # (C)
        rf"\b({choice_pat})\b",             # standalone word boundary
    ]:
        m = re.search(pat, text.upper(), re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


@torch.no_grad()
def predict_example_generation(
    model, tokenizer, example: dict, choices: list[str] | None = None
) -> dict:
    """Generation-based prediction: greedy-decode up to 30 tokens and parse
    the answer letter.  Used for instruction-tuned models (e.g. Gemma4-IT)
    that assign near-zero probability to bare letters as a first response
    token, making log-prob scoring unreliable.
    """
    choices = choices or CHOICES
    raw_prompt = format_mmlu_prompt(example["question"], example["choices"])
    device = next(model.parameters()).device
    prompt_ids = _tokenize_prompt_for_generation(tokenizer, raw_prompt).to(device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        out[0][prompt_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    pred = _parse_choice(generated, choices)
    if pred is None:
        pred = choices[0]
    gold = normalize_answer(example["answer"])
    return {
        "subject": example["subject"],
        "question_id": example["question_id"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold),
        "scores": {},
        "generated": generated,
    }


@torch.no_grad()
def predict_batch_generation(
    model, tokenizer, examples: list[dict], choices: list[str] | None = None
) -> list[dict]:
    """Batched generation: runs multiple examples in one model.generate call.

    Left-pads inputs to a common length so all examples in the batch share
    the same tensor shape.  Gives ~4x throughput over the single-example
    path while producing identical outputs.
    """
    choices = choices or CHOICES
    device = next(model.parameters()).device
    pad_id = tokenizer.eos_token_id or 0

    raw_prompts = [format_mmlu_prompt(ex["question"], ex["choices"]) for ex in examples]
    all_ids = [_tokenize_prompt_for_generation(tokenizer, p)[0] for p in raw_prompts]

    max_len = max(ids.shape[0] for ids in all_ids)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        offset = max_len - ids.shape[0]
        input_ids[i, offset:] = ids
        attention_mask[i, offset:] = 1

    out = model.generate(
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=pad_id,
    )

    results = []
    for i, example in enumerate(examples):
        generated = tokenizer.decode(out[i][max_len:], skip_special_tokens=True).strip()
        pred = _parse_choice(generated, choices)
        if pred is None:
            pred = choices[0]
        gold = normalize_answer(example["answer"])
        results.append({
            "subject": example["subject"],
            "question_id": example["question_id"],
            "gold": gold,
            "pred": pred,
            "correct": int(pred == gold),
            "scores": {},
            "generated": generated,
        })
    return results


def predict_example(model, tokenizer, example: dict, choices: list[str] | None = None) -> dict:
    choices = choices or CHOICES
    raw_prompt = format_mmlu_prompt(example["question"], example["choices"])
    prompt, add_special_tokens = _prepare_prompt(tokenizer, raw_prompt)
    scores = {
        choice: score_choice(model, tokenizer, prompt, choice, add_special_tokens)
        for choice in choices
    }
    pred = max(scores.items(), key=lambda item: item[1])[0]
    gold = normalize_answer(example["answer"])
    return {
        "subject": example["subject"],
        "question_id": example["question_id"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold),
        "scores": scores,
    }
