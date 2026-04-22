from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llm_pruning_mmlu.finetuning.config import MaskPolicyConfig
from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.dispatch import SEMI_STRUCTURED_METHODS, STRUCTURED_METHODS
from llm_pruning_mmlu.pruning.targets import PruningParameter


@dataclass
class MaskEnforcer:
    """Pruning-kind-agnostic mask enforcer.

    Works identically for structured (mlp_channel) and semi-structured (N:M)
    pruning because both produce a dict[str, torch.Tensor] from prune_model and
    both rely on the same apply_masks call.  Extending the SFT runner to
    semi-structured requires only changing the experiment config — no code here
    needs to change.
    """

    parameters: list[PruningParameter]
    masks: dict[str, torch.Tensor] | None
    pruning_kind: str  # "structured" | "semi_structured" | "none" — informational
    preserve_base: bool = True

    @property
    def active(self) -> bool:
        return self.masks is not None and self.preserve_base

    def enforce(self) -> None:
        if self.active:
            apply_masks(self.parameters, self.masks)  # type: ignore[arg-type]

    @classmethod
    def from_prune_result(
        cls,
        parameters: list[PruningParameter],
        masks: dict[str, torch.Tensor] | None,
        pruning_method: str,
        mask_policy_cfg: MaskPolicyConfig,
    ) -> "MaskEnforcer":
        if pruning_method in STRUCTURED_METHODS:
            kind = "structured"
        elif pruning_method in SEMI_STRUCTURED_METHODS:
            kind = "semi_structured"
        else:
            kind = "none"
        return cls(
            parameters=parameters,
            masks=masks,
            pruning_kind=kind,
            preserve_base=mask_policy_cfg.preserve_base_masks,
        )


class MaskEnforcerCallback(TrainerCallback):
    """HuggingFace TrainerCallback that re-zeros pruned weights after each optimizer step.

    on_step_end fires after optimizer.step() and zero_grad(), so masks applied
    here take effect before the next forward pass.
    """

    def __init__(self, enforcer: MaskEnforcer) -> None:
        self._enforcer = enforcer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        self._enforcer.enforce()
        return control
