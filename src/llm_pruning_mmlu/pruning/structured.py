from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from llm_pruning_mmlu.pruning.structured_targets import StructuredGroup
from llm_pruning_mmlu.pruning.targets import PruningParameter


@torch.no_grad()
def _score_groups(groups: list[StructuredGroup]) -> torch.Tensor:
    """Score each group as sum of L2 norms of its member slices.

    Groups are ordered layer-first (as returned by discover_mlp_channel_groups),
    so we batch norm computations per layer — one .norm(dim=1/0) call per weight
    matrix instead of one per channel.
    """
    scores = torch.empty(len(groups), dtype=torch.float32)
    i = 0
    while i < len(groups):
        layer = groups[i].layer_name
        j = i
        while j < len(groups) and groups[j].layer_name == layer:
            j += 1
        layer_groups = groups[i:j]

        gate_param = up_param = down_param = None
        for ts in layer_groups[0].slices:
            if ts.dim == 0 and "gate_proj" in ts.parameter_name:
                gate_param = ts.parameter
            elif ts.dim == 0 and "up_proj" in ts.parameter_name:
                up_param = ts.parameter
            elif ts.dim == 1 and "down_proj" in ts.parameter_name:
                down_param = ts.parameter

        if gate_param is not None and up_param is not None and down_param is not None:
            gate_norms = gate_param.detach().to(device="cpu", dtype=torch.float32).norm(dim=1)
            up_norms = up_param.detach().to(device="cpu", dtype=torch.float32).norm(dim=1)
            down_norms = down_param.detach().to(device="cpu", dtype=torch.float32).norm(dim=0)
            layer_scores = gate_norms + up_norms + down_norms
            # Groups within a layer are created in channel order 0..intermediate_size-1,
            # so layer_scores[k] corresponds directly to layer_groups[k].
            scores[i:j] = layer_scores
        else:
            for k, group in enumerate(layer_groups):
                s = 0.0
                for ts in group.slices:
                    w = ts.parameter.detach().to(device="cpu", dtype=torch.float32)
                    s += float(w[ts.index].norm(2) if ts.dim == 0 else w[:, ts.index].norm(2))
                scores[i + k] = s

        i = j
    return scores


@torch.no_grad()
def compute_structured_masks(
    groups: list[StructuredGroup],
    sparsity: float,
    parameters: list[PruningParameter],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build boolean masks for structured group pruning.

    Returns:
        masks: parameter-name → bool tensor (True = keep), compatible with
               apply_masks so the unstructured path is unchanged.
        group_stats: group-level counts for pruning_stats.json.

    GPU writes are batched per (parameter, dim) pair — one index operation
    zeros all pruned rows/columns for a given weight matrix at once, instead of
    one scalar write per pruned channel. For Llama 3.1 8B at 70% sparsity this
    reduces ~963 K individual CUDA writes to 96 batch ops (~10 000× fewer syncs).
    """
    num_groups = len(groups)
    num_to_prune = int(math.floor(num_groups * (sparsity / 100.0)))

    param_names = {item.name for item in parameters}
    masks: dict[str, torch.Tensor] = {
        item.name: torch.ones_like(item.parameter, dtype=torch.bool)
        for item in parameters
    }

    group_stats: dict[str, Any] = {
        "num_groups_total": num_groups,
        "num_groups_pruned": 0,
        "group_sparsity": 0.0,
        "by_layer": {},
    }

    if num_to_prune <= 0 or num_groups == 0:
        return masks, group_stats

    num_to_prune = min(num_to_prune, num_groups)
    scores = _score_groups(groups)
    prune_indices = torch.argsort(scores, stable=True)[:num_to_prune]
    pruned_set = set(prune_indices.tolist())

    # Collect pruned channel indices grouped by (parameter_name, dim) so we can
    # zero them all in a single batched index operation per weight matrix.
    param_dim_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    layer_pruned: dict[str, int] = defaultdict(int)
    layer_total: dict[str, int] = defaultdict(int)

    for idx, group in enumerate(groups):
        layer_total[group.layer_name] += 1
        if idx in pruned_set:
            layer_pruned[group.layer_name] += 1
            for ts in group.slices:
                if ts.parameter_name in param_names:
                    param_dim_indices[(ts.parameter_name, ts.dim)].append(ts.index)

    # One batch op per weight matrix — vastly fewer GPU syncs than scalar writes.
    for (param_name, dim), indices in param_dim_indices.items():
        mask = masks[param_name]
        idx_t = torch.tensor(indices, dtype=torch.long, device=mask.device)
        if dim == 0:
            mask[idx_t] = False
        else:
            mask[:, idx_t] = False

    group_stats["num_groups_pruned"] = num_to_prune
    group_stats["group_sparsity"] = 100.0 * num_to_prune / num_groups
    group_stats["by_layer"] = {
        ln: {
            "groups_total": layer_total[ln],
            "groups_pruned": layer_pruned[ln],
        }
        for ln in layer_total
    }

    return masks, group_stats
