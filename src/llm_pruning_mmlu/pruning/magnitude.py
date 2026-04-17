from __future__ import annotations

import math

import torch

from llm_pruning_mmlu.pruning.targets import PruningParameter


def _num_to_prune(total: int, sparsity: float) -> int:
    if not 0 <= sparsity <= 100:
        raise ValueError(f"Sparsity must be in [0, 100], got {sparsity}")
    return int(math.floor(total * (sparsity / 100.0)))


def compute_global_magnitude_masks(
    parameters: list[PruningParameter], sparsity: float
) -> dict[str, torch.Tensor]:
    total = sum(item.parameter.numel() for item in parameters)
    prune_count = _num_to_prune(total, sparsity)
    if total == 0:
        return {}
    if prune_count <= 0:
        return {item.name: torch.ones_like(item.parameter, dtype=torch.bool) for item in parameters}
    if prune_count >= total:
        return {
            item.name: torch.zeros_like(item.parameter, dtype=torch.bool) for item in parameters
        }

    flat_abs = torch.cat(
        [
            item.parameter.detach().abs().reshape(-1).to(device="cpu", dtype=torch.float32)
            for item in parameters
        ]
    )
    # Stable exact-count pruning: sort globally and prune the selected flat indices.
    prune_indices = torch.argsort(flat_abs, stable=True)[:prune_count]
    global_mask = torch.ones(total, dtype=torch.bool)
    global_mask[prune_indices] = False

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for item in parameters:
        count = item.parameter.numel()
        mask = global_mask[offset : offset + count].reshape(item.parameter.shape)
        masks[item.name] = mask.to(device=item.parameter.device)
        offset += count
    return masks
