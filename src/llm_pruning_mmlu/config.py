from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    family: str | None = None
    chat_template: bool = False
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceConfig:
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    load_in_4bit: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "mmlu"
    hf_id: str = "cais/mmlu"
    split: str = "test"
    subjects: str | list[str] = "all"
    answer_choices: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    fixture_path: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 4
    max_samples: int | None = None
    subjects: str | list[str] = "all"
    split: str = "test"
    few_shot: int = 0
    scoring_mode: str = "choice_logprob"


@dataclass(frozen=True)
class PruningConfig:
    method: str = "global_magnitude_unstructured"
    structure: str | None = None
    score: str = "l2_norm"
    scope: str = "global"
    nm_n: int | None = None
    nm_m: int | None = None
    block_dim: int = 1
    calibration_samples: int = 128
    target_module_types: list[str] = field(default_factory=lambda: ["Linear"])
    target_parameter_names: list[str] = field(default_factory=lambda: ["weight"])
    exclude_module_name_patterns: list[str] = field(default_factory=lambda: ["lm_head"])
    prune_bias: bool = False
    sparsities: list[float] = field(default_factory=lambda: [0, 50])


@dataclass(frozen=True)
class ReportingConfig:
    save_csv: bool = True
    save_json: bool = True
    save_plot: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    output_root: str
    log_level: str
    save_predictions: bool
    save_pruned_checkpoint: bool
    resume: bool
    device: DeviceConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    pruning: PruningConfig
    reporting: ReportingConfig
    model: ModelConfig | None = None
    models: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def load_config_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = _load_yaml(path)
    merged: dict[str, Any] = {}
    for parent in data.get("inherits", []) or []:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
            if not parent_path.exists():
                parent_path = Path(parent)
        merged = _deep_merge(merged, load_config_dict(parent_path))
    return _deep_merge(merged, data)


def load_model_config(path: str | Path) -> ModelConfig:
    data = load_config_dict(path)
    return ModelConfig(**data["model"])


_NM_METHODS = {"global_magnitude_semi_structured", "wanda_semi_structured"}
_VALID_SEMI_STRUCTURED_SPARSITIES = {0, 50}


def _validate_pruning_config(pruning_cfg: "PruningConfig") -> None:
    if pruning_cfg.method in _NM_METHODS:
        if pruning_cfg.nm_n is None or pruning_cfg.nm_m is None:
            raise ConfigError(
                f"nm_n and nm_m must both be set for method {pruning_cfg.method!r}"
            )
        bad = [s for s in pruning_cfg.sparsities if s not in _VALID_SEMI_STRUCTURED_SPARSITIES]
        if bad:
            raise ConfigError(
                f"For {pruning_cfg.method!r}, sparsities must contain only 0 or 50 "
                f"(the N:M pattern is determined by nm_n/nm_m, not by the sparsity float). "
                f"Got unexpected values: {bad}"
            )


def parse_config(data: dict[str, Any]) -> ExperimentConfig:
    required = ["seed", "output_root", "device", "dataset", "evaluation", "pruning", "reporting"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")
    pruning_cfg = PruningConfig(**data["pruning"])
    _validate_pruning_config(pruning_cfg)
    model = ModelConfig(**data["model"]) if "model" in data else None
    return ExperimentConfig(
        seed=int(data["seed"]),
        output_root=str(data["output_root"]),
        log_level=str(data.get("log_level", "INFO")),
        save_predictions=bool(data.get("save_predictions", True)),
        save_pruned_checkpoint=bool(data.get("save_pruned_checkpoint", False)),
        resume=bool(data.get("resume", True)),
        device=DeviceConfig(**data["device"]),
        dataset=DatasetConfig(**data["dataset"]),
        evaluation=EvaluationConfig(**data["evaluation"]),
        pruning=pruning_cfg,
        reporting=ReportingConfig(**data["reporting"]),
        model=model,
        models=list(data.get("models", [])),
        raw=data,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    return parse_config(load_config_dict(path))


def dump_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
