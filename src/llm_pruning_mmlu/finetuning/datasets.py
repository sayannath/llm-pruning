from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from llm_pruning_mmlu.data.mmlu import load_mmlu
from llm_pruning_mmlu.data.prompting import format_mmlu_prompt, normalize_answer
from llm_pruning_mmlu.finetuning.config import DatasetSplitConfig

_RESPONSE_TRIGGER = "Answer:"


class MmluSftDataset(Dataset):
    """MMLU examples formatted for causal-LM supervised fine-tuning.

    Each item has input_ids, attention_mask, and labels.  Labels are -100
    (ignored) for every prompt token so the loss is computed only on the
    answer letter, matching the evaluation scoring mode.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        max_seq_length: int,
    ) -> None:
        self._items = _tokenize_examples(examples, tokenizer, max_seq_length)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]


def _tokenize_examples(
    examples: list[dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> list[dict[str, torch.Tensor]]:
    items = []
    for ex in examples:
        prompt = format_mmlu_prompt(ex["question"], ex["choices"])
        answer = normalize_answer(ex["answer"])
        full_text = prompt + " " + answer

        full_enc = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        # Tokenize the answer suffix without BOS so we can count answer tokens
        # independently of how the tokenizer handles the space-letter boundary.
        # Tokenizing `prompt + " "` is unreliable because many tokenizers merge
        # " A" into a single token (▁A), making prompt_len == full_len and
        # masking all labels — producing NaN loss logged as 0 in wandb.
        answer_enc = tokenizer(
            " " + answer,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        answer_len = answer_enc["input_ids"].shape[1]
        prompt_len = max(0, len(input_ids) - answer_len)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        items.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return items


def load_sft_dataset(
    cfg: DatasetSplitConfig,
    tokenizer,
    max_seq_length: int,
) -> MmluSftDataset:
    examples = load_mmlu(
        hf_id=cfg.hf_id,
        split=cfg.split,
        max_samples=cfg.max_samples,
    )
    return MmluSftDataset(examples, tokenizer, max_seq_length)
