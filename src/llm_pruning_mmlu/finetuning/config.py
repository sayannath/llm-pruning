from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WandbConfig:
    project: str = "sparse-sft"
    enabled: bool = True


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class MaskPolicyConfig:
    preserve_base_masks: bool = True
    mask_lora_pruned_channels: bool = False


@dataclass(frozen=True)
class DatasetSplitConfig:
    hf_id: str = "cais/mmlu"
    split: str = "auxiliary_train"
    max_samples: int | None = None


@dataclass(frozen=True)
class FinetuningConfig:
    method: str = "lora"
    train_dataset: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    validation_dataset: DatasetSplitConfig = field(
        default_factory=lambda: DatasetSplitConfig(split="validation", max_samples=512)
    )
    max_seq_length: int = 1024
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    mask_policy: MaskPolicyConfig = field(default_factory=MaskPolicyConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _parse_dataset_split(d: dict[str, Any]) -> DatasetSplitConfig:
    return DatasetSplitConfig(
        hf_id=d.get("hf_id", "cais/mmlu"),
        split=d.get("split", "auxiliary_train"),
        max_samples=d.get("max_samples"),
    )


def _parse_lora(d: dict[str, Any]) -> LoraConfig:
    modules = d.get("target_modules")
    return LoraConfig(
        r=int(d.get("r", 16)),
        alpha=int(d.get("alpha", 32)),
        dropout=float(d.get("dropout", 0.05)),
        target_modules=tuple(modules) if modules is not None else LoraConfig().target_modules,
    )


def _parse_mask_policy(d: dict[str, Any]) -> MaskPolicyConfig:
    return MaskPolicyConfig(
        preserve_base_masks=bool(d.get("preserve_base_masks", True)),
        mask_lora_pruned_channels=bool(d.get("mask_lora_pruned_channels", False)),
    )


def _parse_wandb(d: dict[str, Any]) -> WandbConfig:
    return WandbConfig(
        project=str(d.get("project", "sparse-sft")),
        enabled=bool(d.get("enabled", True)),
    )


def parse_finetuning_config(data: dict[str, Any]) -> FinetuningConfig:
    ft = data.get("finetuning", data)
    return FinetuningConfig(
        method=str(ft.get("method", "lora")),
        train_dataset=_parse_dataset_split(ft.get("train_dataset", {})),
        validation_dataset=_parse_dataset_split(
            ft.get("validation_dataset", {"split": "validation", "max_samples": 512})
        ),
        max_seq_length=int(ft.get("max_seq_length", 1024)),
        epochs=int(ft.get("epochs", 1)),
        learning_rate=float(ft.get("learning_rate", 2e-4)),
        batch_size=int(ft.get("batch_size", 1)),
        gradient_accumulation_steps=int(ft.get("gradient_accumulation_steps", 16)),
        warmup_ratio=float(ft.get("warmup_ratio", 0.03)),
        weight_decay=float(ft.get("weight_decay", 0.0)),
        bf16=bool(ft.get("bf16", True)),
        gradient_checkpointing=bool(ft.get("gradient_checkpointing", True)),
        lora=_parse_lora(ft.get("lora", {})),
        mask_policy=_parse_mask_policy(ft.get("mask_policy", {})),
        wandb=_parse_wandb(ft.get("wandb", {})),
    )


def load_finetuning_config(path: str | Path) -> FinetuningConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Finetuning config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return parse_finetuning_config(data)
