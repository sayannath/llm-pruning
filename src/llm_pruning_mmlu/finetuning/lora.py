from __future__ import annotations

import torch
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model

from llm_pruning_mmlu.finetuning.config import LoraConfig


def _resolve_target_modules(model: torch.nn.Module, target_modules: list[str]) -> list[str]:
    # Gemma4 wraps every projection in Gemma4ClippableLinear which contains a
    # plain Linear at .linear. PEFT can't wrap the outer class, so we target
    # the inner submodule: "q_proj" -> "q_proj.linear".
    for module in model.modules():
        if type(module).__name__ == "Gemma4ClippableLinear":
            return [f"{t}.linear" for t in target_modules]
    return target_modules


def attach_lora(model: torch.nn.Module, cfg: LoraConfig) -> torch.nn.Module:
    target_modules = _resolve_target_modules(model, list(cfg.target_modules))
    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model
