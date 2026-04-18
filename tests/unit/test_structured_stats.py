from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.config import PruningConfig
from llm_pruning_mmlu.pruning.dispatch import prune_model
from llm_pruning_mmlu.pruning.structured import compute_structured_masks
from llm_pruning_mmlu.pruning.structured_targets import discover_mlp_channel_groups
from llm_pruning_mmlu.pruning.targets import find_pruning_parameters


class _SwiGluMlp(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _MlpModel(nn.Module):
    def __init__(self, num_layers: int = 2, intermediate: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [_SwiGluMlp(intermediate=intermediate) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(4, 16, bias=False)


# ---------------------------------------------------------------------------
# group_stats fields
# ---------------------------------------------------------------------------

def test_zero_sparsity_group_stats():
    model = _MlpModel()
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    _, gs = compute_structured_masks(groups, 0.0, params)
    assert gs["num_groups_total"] == 16
    assert gs["num_groups_pruned"] == 0
    assert gs["group_sparsity"] == 0.0


def test_fifty_sparsity_group_stats():
    model = _MlpModel()
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    _, gs = compute_structured_masks(groups, 50.0, params)
    assert gs["num_groups_pruned"] == 8
    assert abs(gs["group_sparsity"] - 50.0) < 1e-3


def test_by_layer_keys_present():
    model = _MlpModel(num_layers=3, intermediate=6)
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    _, gs = compute_structured_masks(groups, 33.0, params)
    assert len(gs["by_layer"]) == 3
    for layer_name, layer_gs in gs["by_layer"].items():
        assert "groups_total" in layer_gs
        assert "groups_pruned" in layer_gs
        assert layer_gs["groups_total"] == 6


# ---------------------------------------------------------------------------
# dispatcher stats
# ---------------------------------------------------------------------------

def test_dispatcher_stats_include_group_fields():
    model = _MlpModel()
    cfg = PruningConfig(
        method="global_magnitude_structured",
        structure="mlp_channel",
        exclude_module_name_patterns=["lm_head"],
    )
    _, stats = prune_model(model, cfg, 25.0)
    assert "num_groups_total" in stats
    assert "num_groups_pruned" in stats
    assert "group_sparsity" in stats
    assert "by_layer" in stats


def test_dispatcher_parameter_sparsity_matches_reality():
    model = _MlpModel()
    cfg = PruningConfig(
        method="global_magnitude_structured",
        structure="mlp_channel",
        exclude_module_name_patterns=["lm_head"],
    )
    _, stats = prune_model(model, cfg, 50.0)
    # parameter sparsity must be > 0 after pruning half the groups
    assert stats["sparsity"] > 0.0
    # reported nonzero + zero must equal total
    assert stats["nonzero"] + stats["zero"] == stats["total"]


def test_group_sparsity_reported_equals_requested_at_exact_count():
    model = _MlpModel(num_layers=1, intermediate=10)
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    # 10 groups, 50% → 5 pruned exactly
    _, gs = compute_structured_masks(groups, 50.0, params)
    assert gs["num_groups_pruned"] == 5
    assert abs(gs["group_sparsity"] - 50.0) < 1e-3
