from __future__ import annotations

import torch

from llm_pruning_mmlu.pruning.targets import PruningParameter


@torch.no_grad()
def apply_masks(parameters: list[PruningParameter], masks: dict[str, torch.Tensor]) -> None:
    for item in parameters:
        mask = masks[item.name].to(device=item.parameter.device)
        item.parameter.mul_(mask.to(dtype=item.parameter.dtype))
