from __future__ import annotations

import logging
from typing import Any

import torch

from llm_pruning_mmlu.pruning.targets import PruningParameter

_log = logging.getLogger("llm_pruning_mmlu")

_SUPPORTED_PATTERNS: frozenset[tuple[int, int]] = frozenset({(2, 4), (4, 8)})


def _check_block_dim(block_dim: int) -> None:
    if block_dim != 1:
        raise ValueError(
            f"Only block_dim=1 (row-wise blocks over input dimension) is supported "
            f"in phase 1; got block_dim={block_dim}"
        )


@torch.no_grad()
def compute_nm_magnitude_masks(
    parameters: list[PruningParameter],
    n: int,
    m: int,
    block_dim: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build N:M boolean masks for all parameters using local magnitude pruning.

    For each contiguous block of M weights along the input dimension (dim=1),
    the N smallest-magnitude weights are pruned (set False in the mask).
    Remainder weights at the end of rows where in_features % M != 0 are kept.
    """
    _check_block_dim(block_dim)
    if not (0 < n < m):
        raise ValueError(f"N:M pattern requires 0 < n < m; got n={n}, m={m}")
    if (n, m) not in _SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unsupported N:M pattern ({n}:{m}). "
            f"Supported: {sorted(_SUPPORTED_PATTERNS)}"
        )

    masks: dict[str, torch.Tensor] = {}
    total_blocks = 0
    total_remainder_weights = 0
    total_pruned = 0

    for item in parameters:
        w = item.parameter

        if w.ndim != 2:
            masks[item.name] = torch.ones_like(w, dtype=torch.bool)
            continue

        out_features, in_features = w.shape
        num_blocks_per_row = in_features // m
        remainder = in_features % m

        if num_blocks_per_row == 0:
            _log.warning(
                "Layer %s has in_features=%d < m=%d; "
                "entire layer is remainder — nothing pruned",
                item.name,
                in_features,
                m,
            )
            masks[item.name] = torch.ones_like(w, dtype=torch.bool)
            total_remainder_weights += w.numel()
            continue

        mask = torch.ones(out_features, in_features, dtype=torch.bool)
        complete_end = num_blocks_per_row * m

        w_cpu = w.detach().to(device="cpu", dtype=torch.float32)
        # [out_features, num_blocks_per_row, m]
        w_blocks = w_cpu[:, :complete_end].reshape(out_features, num_blocks_per_row, m)

        # argsort ascending along block dimension: index 0 = smallest magnitude
        order = w_blocks.abs().argsort(dim=2, stable=True)  # [out, num_blocks, m]

        # Mark positions of the n smallest-magnitude weights per block for pruning
        prune_local = torch.zeros(out_features, num_blocks_per_row, m, dtype=torch.bool)
        prune_local.scatter_(2, order[:, :, :n], True)

        mask[:, :complete_end] = (~prune_local).reshape(out_features, complete_end)
        masks[item.name] = mask.to(device=w.device)

        layer_blocks = out_features * num_blocks_per_row
        total_blocks += layer_blocks
        total_remainder_weights += out_features * remainder
        total_pruned += layer_blocks * n

    total_weights_in_complete_blocks = total_blocks * m
    nm_sparsity = (
        100.0 * total_pruned / total_weights_in_complete_blocks
        if total_weights_in_complete_blocks > 0
        else 0.0
    )

    stats: dict[str, Any] = {
        "nm_n": n,
        "nm_m": m,
        "block_dim": block_dim,
        "num_blocks_total": total_blocks,
        "num_complete_blocks": total_blocks,
        "num_remainder_weights": total_remainder_weights,
        "num_weights_pruned_by_nm": total_pruned,
        "nm_sparsity": nm_sparsity,
    }
    return masks, stats


@torch.no_grad()
def validate_nm_mask(
    parameter: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    m: int,
    block_dim: int = 1,
) -> dict[str, Any]:
    """Verify that every complete N:M block has exactly n zeros.

    Returns a dict with violation counts. Raises on shape mismatch or
    unsupported block_dim.
    """
    _check_block_dim(block_dim)
    if parameter.shape != mask.shape:
        raise ValueError(
            f"Parameter shape {tuple(parameter.shape)} != mask shape {tuple(mask.shape)}"
        )

    if parameter.ndim != 2:
        return {"violations": 0, "blocks_checked": 0, "status": "skipped_non2d"}

    out_features, in_features = parameter.shape
    num_blocks_per_row = in_features // m

    if num_blocks_per_row == 0:
        remainder_violations = int((~mask.to(device="cpu")).sum().item())
        return {
            "violations": remainder_violations,
            "blocks_checked": 0,
            "remainder_violations": remainder_violations,
            "status": "ok" if remainder_violations == 0 else "remainder_violation",
        }

    complete_end = num_blocks_per_row * m
    mask_cpu = mask.to(device="cpu")

    # zeros in complete block region: True where mask=False (weight pruned)
    zeros_complete = ~mask_cpu[:, :complete_end]
    zeros_blocks = zeros_complete.reshape(out_features, num_blocks_per_row, m)
    zeros_per_block = zeros_blocks.sum(dim=2)  # [out, num_blocks]
    block_violations = int((zeros_per_block != n).sum().item())

    remainder_violations = 0
    if in_features % m != 0:
        remainder_violations = int((~mask_cpu[:, complete_end:]).sum().item())

    total_violations = block_violations + remainder_violations
    return {
        "violations": total_violations,
        "block_violations": block_violations,
        "remainder_violations": remainder_violations,
        "blocks_checked": out_features * num_blocks_per_row,
        "status": "ok" if total_violations == 0 else "failed",
    }
