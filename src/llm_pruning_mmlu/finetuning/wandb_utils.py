from __future__ import annotations

import os
from typing import Any

from llm_pruning_mmlu.finetuning.config import WandbConfig


class WandbRun:
    """Thin wrapper around a wandb run.

    All methods are no-ops when enabled=False, so the training loop never
    needs to branch on wandb availability.
    """

    def __init__(self, enabled: bool, _run: Any = None) -> None:
        self._enabled = enabled
        self._run = _run

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._enabled or self._run is None:
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        import wandb
        for key, val in metrics.items():
            wandb.run.summary[key] = val  # type: ignore[union-attr]

    def finish(self) -> None:
        if not self._enabled or self._run is None:
            return
        import wandb
        wandb.finish()


def init_wandb(
    cfg: WandbConfig,
    run_name: str,
    config_dict: dict[str, Any],
) -> WandbRun:
    """Initialise a wandb run and return a WandbRun handle.

    If cfg.enabled is False or WANDB_MODE=disabled, returns a no-op run
    so the rest of the codebase never needs to check for wandb availability.
    """
    if not cfg.enabled or os.environ.get("WANDB_MODE") == "disabled":
        return WandbRun(enabled=False)

    try:
        import wandb
    except ImportError:
        return WandbRun(enabled=False)

    run = wandb.init(
        project=cfg.project,
        name=run_name,
        config=config_dict,
        reinit=True,
    )
    return WandbRun(enabled=True, _run=run)


def wandb_run_name(model_name: str, sparsity: float, method: str = "lora") -> str:
    return f"{model_name}_sparsity_{int(sparsity):03d}_{method}"


def report_to_flag(cfg: WandbConfig) -> str:
    if cfg.enabled and os.environ.get("WANDB_MODE") != "disabled":
        return "wandb"
    return "none"
