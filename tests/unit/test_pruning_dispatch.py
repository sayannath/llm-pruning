from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.config import PruningConfig
from llm_pruning_mmlu.pruning.dispatch import prune_model


class _SwiGluMlp(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_SwiGluMlp(), _SwiGluMlp()])
        self.lm_head = nn.Linear(4, 16, bias=False)


class _PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.lm_head = nn.Linear(4, 16, bias=False)


_UNSTRUCTURED_CFG = PruningConfig(
    method="global_magnitude_unstructured",
    exclude_module_name_patterns=["lm_head"],
)

_STRUCTURED_CFG = PruningConfig(
    method="global_magnitude_structured",
    structure="mlp_channel",
    exclude_module_name_patterns=["lm_head"],
)


# ---------------------------------------------------------------------------
# Unstructured dispatch
# ---------------------------------------------------------------------------

def test_unstructured_zero_sparsity_no_change():
    model = _PlainModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prune_model(model, _UNSTRUCTURED_CFG, 0.0)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n])


def test_unstructured_stats_keys():
    model = _PlainModel()
    _, stats = prune_model(model, _UNSTRUCTURED_CFG, 50.0)
    for key in ("total", "nonzero", "sparsity", "layers", "method", "structure"):
        assert key in stats
    assert stats["method"] == "global_magnitude_unstructured"
    assert stats["structure"] is None


def test_unstructured_no_group_stats():
    model = _PlainModel()
    _, stats = prune_model(model, _UNSTRUCTURED_CFG, 50.0)
    assert "num_groups_total" not in stats


# ---------------------------------------------------------------------------
# Structured dispatch
# ---------------------------------------------------------------------------

def test_structured_zero_sparsity_no_change():
    model = _MlpModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prune_model(model, _STRUCTURED_CFG, 0.0)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n])


def test_structured_stats_keys():
    model = _MlpModel()
    _, stats = prune_model(model, _STRUCTURED_CFG, 20.0)
    for key in (
        "total", "nonzero", "sparsity", "method", "structure",
        "num_groups_total", "num_groups_pruned", "group_sparsity",
    ):
        assert key in stats
    assert stats["method"] == "global_magnitude_structured"
    assert stats["structure"] == "mlp_channel"


def test_structured_group_count():
    model = _MlpModel()  # 2 layers × 8 intermediate = 16 groups
    _, stats = prune_model(model, _STRUCTURED_CFG, 0.0)
    assert stats["num_groups_total"] == 16


def test_structured_fifty_sparsity_prunes_half():
    model = _MlpModel()
    _, stats = prune_model(model, _STRUCTURED_CFG, 50.0)
    assert stats["num_groups_pruned"] == stats["num_groups_total"] // 2


def test_structured_achieves_nonzero_parameter_sparsity():
    model = _MlpModel()
    _, stats = prune_model(model, _STRUCTURED_CFG, 50.0)
    assert stats["sparsity"] > 0.0


def test_unknown_method_raises():
    model = _PlainModel()
    cfg = PruningConfig(method="invalid_method")
    try:
        prune_model(model, cfg, 0.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_unknown_structure_raises():
    model = _MlpModel()
    cfg = PruningConfig(method="global_magnitude_structured", structure="unknown_structure")
    try:
        prune_model(model, cfg, 20.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Semi-structured dispatch
# ---------------------------------------------------------------------------

_SEMI_24_CFG = PruningConfig(
    method="global_magnitude_semi_structured",
    structure="nm_2_4",
    nm_n=2,
    nm_m=4,
    block_dim=1,
    exclude_module_name_patterns=["lm_head"],
)

_SEMI_48_CFG = PruningConfig(
    method="global_magnitude_semi_structured",
    structure="nm_4_8",
    nm_n=4,
    nm_m=8,
    block_dim=1,
    exclude_module_name_patterns=["lm_head"],
)


class _SmallLinearModel(nn.Module):
    """Simple model with in_features divisible by 8 so both 2:4 and 4:8 work."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)


def test_semi_structured_24_zero_sparsity_no_change():
    model = _SmallLinearModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prune_model(model, _SEMI_24_CFG, 0.0)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"{n} changed at sparsity=0"


def test_semi_structured_48_zero_sparsity_no_change():
    model = _SmallLinearModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prune_model(model, _SEMI_48_CFG, 0.0)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"{n} changed at sparsity=0"


def test_semi_structured_24_stats_keys():
    model = _SmallLinearModel()
    _, stats = prune_model(model, _SEMI_24_CFG, 50.0)
    for key in (
        "total", "nonzero", "sparsity",
        "nm_n", "nm_m", "block_dim",
        "num_blocks_total", "num_complete_blocks",
        "num_remainder_weights", "nm_sparsity",
        "method", "structure",
    ):
        assert key in stats, f"Missing key: {key}"
    assert stats["nm_n"] == 2
    assert stats["nm_m"] == 4
    assert stats["method"] == "global_magnitude_semi_structured"


def test_semi_structured_48_stats_keys():
    model = _SmallLinearModel()
    _, stats = prune_model(model, _SEMI_48_CFG, 50.0)
    assert stats["nm_n"] == 4
    assert stats["nm_m"] == 8


def test_semi_structured_achieves_nonzero_sparsity():
    model = _SmallLinearModel()
    _, stats = prune_model(model, _SEMI_24_CFG, 50.0)
    assert stats["sparsity"] > 0.0


def test_semi_structured_nm_sparsity_is_50():
    model = _SmallLinearModel()
    _, stats = prune_model(model, _SEMI_24_CFG, 50.0)
    # fc1 and fc2 both have in_features=8 which is divisible by 4 → no remainder
    assert abs(stats["nm_sparsity"] - 50.0) < 1e-4


def test_semi_structured_no_group_stats():
    model = _SmallLinearModel()
    _, stats = prune_model(model, _SEMI_24_CFG, 50.0)
    assert "num_groups_total" not in stats


def test_semi_structured_lm_head_excluded():
    """lm_head weights must be untouched after semi-structured pruning."""
    model = _SmallLinearModel()
    lm_head_before = model.lm_head.weight.detach().clone()
    prune_model(model, _SEMI_24_CFG, 50.0)
    assert torch.equal(model.lm_head.weight, lm_head_before)
