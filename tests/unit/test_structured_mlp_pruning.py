from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.structured import compute_structured_masks
from llm_pruning_mmlu.pruning.structured_targets import StructuredGroup, discover_mlp_channel_groups
from llm_pruning_mmlu.pruning.targets import find_pruning_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SwiGluMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _TwoLayerModel(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [_SwiGluMlp(hidden, intermediate), _SwiGluMlp(hidden, intermediate)]
        )
        self.lm_head = nn.Linear(hidden, 16, bias=False)


def _make_model(hidden: int = 4, intermediate: int = 8) -> _TwoLayerModel:
    model = _TwoLayerModel(hidden, intermediate)
    nn.init.normal_(model.layers[0].gate_proj.weight)
    nn.init.normal_(model.layers[0].up_proj.weight)
    nn.init.normal_(model.layers[0].down_proj.weight)
    nn.init.normal_(model.layers[1].gate_proj.weight)
    nn.init.normal_(model.layers[1].up_proj.weight)
    nn.init.normal_(model.layers[1].down_proj.weight)
    return model


# ---------------------------------------------------------------------------
# Group discovery
# ---------------------------------------------------------------------------

def test_discover_finds_correct_group_count():
    model = _make_model(hidden=4, intermediate=8)
    groups = discover_mlp_channel_groups(model)
    # 2 layers × 8 intermediate channels = 16 groups
    assert len(groups) == 16


def test_discover_excludes_lm_head():
    model = _make_model()
    groups = discover_mlp_channel_groups(model)
    for g in groups:
        assert "lm_head" not in g.layer_name


def test_discover_group_has_three_slices():
    model = _make_model()
    groups = discover_mlp_channel_groups(model)
    for g in groups:
        assert len(g.slices) == 3
        dims = {ts.dim for ts in g.slices}
        assert dims == {0, 1}


def test_discover_slices_point_to_correct_parameter_names():
    model = _make_model()
    groups = discover_mlp_channel_groups(model)
    for g in groups:
        names = {ts.parameter_name for ts in g.slices}
        assert any("gate_proj" in n for n in names)
        assert any("up_proj" in n for n in names)
        assert any("down_proj" in n for n in names)


def test_discover_channel_index_matches_slice_index():
    model = _make_model(hidden=4, intermediate=6)
    groups = discover_mlp_channel_groups(model)
    layer_groups = [g for g in groups if g.layer_name == "layers.0"]
    for j, g in enumerate(layer_groups):
        for ts in g.slices:
            assert ts.index == j


# ---------------------------------------------------------------------------
# Mask correctness
# ---------------------------------------------------------------------------

def test_zero_sparsity_keeps_all_weights():
    model = _make_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    masks, _ = compute_structured_masks(groups, 0.0, params)
    apply_masks(params, masks)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"{n} changed at sparsity=0"


def test_fifty_percent_prunes_exactly_half_groups():
    model = _make_model(hidden=4, intermediate=8)
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    masks, group_stats = compute_structured_masks(groups, 50.0, params)
    total = group_stats["num_groups_total"]
    pruned = group_stats["num_groups_pruned"]
    assert pruned == total // 2


def test_masks_preserve_tensor_shapes():
    model = _make_model()
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    masks, _ = compute_structured_masks(groups, 50.0, params)
    for item in params:
        assert masks[item.name].shape == item.parameter.shape


def test_channel_masks_are_consistent_across_gate_up_down():
    """Pruning channel j must zero gate_proj row j, up_proj row j, down_proj col j."""
    model = _make_model(hidden=4, intermediate=8)
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    masks, _ = compute_structured_masks(groups, 50.0, params)
    apply_masks(params, masks)

    for layer_idx in range(2):
        mlp = model.layers[layer_idx]
        gate_w = mlp.gate_proj.weight  # [intermediate, hidden]
        up_w = mlp.up_proj.weight
        down_w = mlp.down_proj.weight  # [hidden, intermediate]

        for j in range(gate_w.shape[0]):
            gate_zero = gate_w[j].abs().sum() == 0
            up_zero = up_w[j].abs().sum() == 0
            down_zero = down_w[:, j].abs().sum() == 0
            # All three must agree: either all zeroed or all kept.
            assert gate_zero == up_zero == down_zero, (
                f"Layer {layer_idx} channel {j}: gate_zero={gate_zero}, "
                f"up_zero={up_zero}, down_zero={down_zero}"
            )


def test_global_selection_prunes_lowest_scoring_groups():
    """The dispatcher must choose globally lowest-magnitude groups, not per-layer."""
    model = _make_model(hidden=4, intermediate=4)
    with torch.no_grad():
        # Make layer 0 channels clearly weaker than layer 1.
        model.layers[0].gate_proj.weight.fill_(0.01)
        model.layers[0].up_proj.weight.fill_(0.01)
        model.layers[0].down_proj.weight.fill_(0.01)
        model.layers[1].gate_proj.weight.fill_(10.0)
        model.layers[1].up_proj.weight.fill_(10.0)
        model.layers[1].down_proj.weight.fill_(10.0)

    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    # Prune 50% of 8 total groups = 4 groups. All 4 should come from layer 0.
    masks, _ = compute_structured_masks(groups, 50.0, params)
    apply_masks(params, masks)

    layer0_gate = model.layers[0].gate_proj.weight
    layer1_gate = model.layers[1].gate_proj.weight
    # At least some of layer 0 should be zeroed, none of layer 1.
    assert (layer0_gate == 0).any()
    assert not (layer1_gate == 0).any()


def test_hundred_percent_sparsity_zeros_all_mlp_weights():
    model = _make_model()
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    groups = discover_mlp_channel_groups(model)
    masks, _ = compute_structured_masks(groups, 100.0, params)
    apply_masks(params, masks)
    for item in params:
        assert item.parameter.abs().sum() == 0, f"{item.name} not fully zeroed"
