from __future__ import annotations

from typing import Any

import torch

from llm_pruning_mmlu.config import PruningConfig
from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.magnitude import compute_global_magnitude_masks
from llm_pruning_mmlu.pruning.semi_structured import compute_nm_magnitude_masks, validate_nm_mask
from llm_pruning_mmlu.pruning.stats import pruning_stats
from llm_pruning_mmlu.pruning.structured import compute_structured_masks
from llm_pruning_mmlu.pruning.structured_targets import discover_mlp_channel_groups
from llm_pruning_mmlu.pruning.targets import PruningParameter, find_pruning_parameters
from llm_pruning_mmlu.pruning.wanda import compute_nm_wanda_masks

_STRUCTURED_METHODS = {"global_magnitude_structured"}
_UNSTRUCTURED_METHODS = {"global_magnitude_unstructured"}
_SEMI_STRUCTURED_METHODS = {"global_magnitude_semi_structured", "wanda_semi_structured"}


def prune_model(
    model: torch.nn.Module,
    pruning_config: PruningConfig,
    sparsity: float,
    tokenizer=None,
) -> tuple[list[PruningParameter], dict[str, Any], dict[str, torch.Tensor] | None]:
    """Apply pruning to *model* in place and return (parameters, stats, masks).

    parameters: the PruningParameter list; keep alive while the model is in
                use, then set to None to release parameter refs.
    stats: pruning_stats-compatible dict, extended with group-level fields
           for structured runs.
    masks: boolean weight masks for structured and semi-structured runs
           (True = keep); None for unstructured. Used by the SFT training loop
           to re-zero pruned weights after every optimizer step via apply_masks.
           Extending to semi-structured SFT is zero-cost — the same masks dict
           and apply_masks call work for both pruning kinds.
    """
    parameters = find_pruning_parameters(
        model,
        target_module_types=pruning_config.target_module_types,
        target_parameter_names=pruning_config.target_parameter_names,
        exclude_module_name_patterns=pruning_config.exclude_module_name_patterns,
        prune_bias=pruning_config.prune_bias,
    )

    method = pruning_config.method

    if method in _UNSTRUCTURED_METHODS:
        return _prune_unstructured(parameters, pruning_config, sparsity)
    elif method in _STRUCTURED_METHODS:
        return _prune_structured(model, parameters, pruning_config, sparsity)
    elif method == "global_magnitude_semi_structured":
        return _prune_semi_structured(parameters, pruning_config, sparsity)
    elif method == "wanda_semi_structured":
        return _prune_wanda_semi_structured(
            parameters, pruning_config, sparsity, model=model, tokenizer=tokenizer
        )
    else:
        raise ValueError(
            f"Unknown pruning method: {method!r}. "
            f"Supported: {sorted(_UNSTRUCTURED_METHODS | _STRUCTURED_METHODS | _SEMI_STRUCTURED_METHODS)}"
        )


STRUCTURED_METHODS = _STRUCTURED_METHODS
SEMI_STRUCTURED_METHODS = _SEMI_STRUCTURED_METHODS


def _prune_unstructured(
    parameters: list[PruningParameter],
    pruning_config: PruningConfig,
    sparsity: float,
) -> tuple[list[PruningParameter], dict[str, Any], None]:
    if sparsity > 0:
        masks = compute_global_magnitude_masks(parameters, sparsity)
        apply_masks(parameters, masks)

    stats = pruning_stats(parameters)
    stats.update(
        {
            "method": pruning_config.method,
            "structure": None,
            "score": pruning_config.score,
            "scope": pruning_config.scope,
        }
    )
    return parameters, stats, None


def _prune_structured(
    model: torch.nn.Module,
    parameters: list[PruningParameter],
    pruning_config: PruningConfig,
    sparsity: float,
) -> tuple[list[PruningParameter], dict[str, Any], dict[str, torch.Tensor] | None]:
    structure = pruning_config.structure

    if structure == "mlp_channel":
        groups = discover_mlp_channel_groups(model)
    else:
        raise ValueError(
            f"Unknown structured pruning structure: {structure!r}. "
            "Supported: ['mlp_channel']"
        )

    group_stats: dict[str, Any]
    masks: dict[str, torch.Tensor] | None = None
    if sparsity > 0 and groups:
        masks, group_stats = compute_structured_masks(groups, sparsity, parameters)
        apply_masks(parameters, masks)
    else:
        group_stats = {
            "num_groups_total": len(groups),
            "num_groups_pruned": 0,
            "group_sparsity": 0.0,
            "by_layer": {},
        }

    stats = pruning_stats(parameters)
    stats.update(
        {
            "method": pruning_config.method,
            "structure": structure,
            "score": pruning_config.score,
            "scope": pruning_config.scope,
            "num_groups_total": group_stats["num_groups_total"],
            "num_groups_pruned": group_stats["num_groups_pruned"],
            "group_sparsity": group_stats["group_sparsity"],
            "by_layer": group_stats["by_layer"],
        }
    )
    return parameters, stats, masks


def _prune_wanda_semi_structured(
    parameters: list[PruningParameter],
    pruning_config: PruningConfig,
    sparsity: float,
    model: torch.nn.Module | None = None,
    tokenizer=None,
) -> tuple[list[PruningParameter], dict[str, Any], dict[str, torch.Tensor] | None]:
    n = pruning_config.nm_n
    m = pruning_config.nm_m
    block_dim = pruning_config.block_dim
    calibration_samples = pruning_config.calibration_samples

    nm_masks: dict[str, torch.Tensor] | None = None
    if sparsity > 0:
        nm_masks, nm_stats = compute_nm_wanda_masks(
            parameters, n, m, block_dim,
            model=model, tokenizer=tokenizer,
            calibration_samples=calibration_samples,
        )
        for item in parameters:
            result = validate_nm_mask(item.parameter, nm_masks[item.name], n, m, block_dim)
            if result["violations"] > 0:
                raise RuntimeError(
                    f"N:M mask validation failed for {item.name}: "
                    f"{result['violations']} violation(s)"
                )
        apply_masks(parameters, nm_masks)
    else:
        nm_stats = {
            "nm_n": n, "nm_m": m, "block_dim": block_dim,
            "num_blocks_total": 0, "num_complete_blocks": 0,
            "num_remainder_weights": 0, "num_weights_pruned_by_nm": 0,
            "nm_sparsity": 0.0,
            "wanda_calibration_samples": calibration_samples,
            "wanda_modules_with_norms": 0,
        }

    stats = pruning_stats(parameters)
    stats.update({
        "method": pruning_config.method,
        "structure": pruning_config.structure,
        "score": pruning_config.score,
        "scope": pruning_config.scope,
        **nm_stats,
    })
    return parameters, stats, nm_masks


def _prune_semi_structured(
    parameters: list[PruningParameter],
    pruning_config: PruningConfig,
    sparsity: float,
) -> tuple[list[PruningParameter], dict[str, Any], dict[str, torch.Tensor] | None]:
    n = pruning_config.nm_n
    m = pruning_config.nm_m
    block_dim = pruning_config.block_dim

    nm_masks: dict[str, torch.Tensor] | None = None
    if sparsity > 0:
        # sparsity is a trigger (0 = no-op, non-zero = apply N:M pattern).
        # The actual achieved sparsity is determined by n and m, not this float.
        nm_masks, nm_stats = compute_nm_magnitude_masks(parameters, n, m, block_dim)
        for item in parameters:
            result = validate_nm_mask(item.parameter, nm_masks[item.name], n, m, block_dim)
            if result["violations"] > 0:
                raise RuntimeError(
                    f"N:M mask validation failed for {item.name}: "
                    f"{result['violations']} violation(s)"
                )
        apply_masks(parameters, nm_masks)
    else:
        nm_stats = {
            "nm_n": n,
            "nm_m": m,
            "block_dim": block_dim,
            "num_blocks_total": 0,
            "num_complete_blocks": 0,
            "num_remainder_weights": 0,
            "num_weights_pruned_by_nm": 0,
            "nm_sparsity": 0.0,
        }

    stats = pruning_stats(parameters)
    stats.update(
        {
            "method": pruning_config.method,
            "structure": pruning_config.structure,
            "score": pruning_config.score,
            "scope": pruning_config.scope,
            **nm_stats,
        }
    )
    return parameters, stats, nm_masks
