"""Tests for MaskEnforcer and MaskEnforcerCallback
(src/llm_pruning_mmlu/finetuning/mask_policy.py).

Covers:
- pruning_kind classification for structured / semi-structured / unstructured
- active property
- enforce() re-zeros pruned weights
- enforce() is no-op when masks is None or preserve_base=False
- MaskEnforcerCallback.on_step_end calls enforce
- Works with structured (mlp_channel) masks
- Works with semi-structured (2:4 and Wanda) masks
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock

from llm_pruning_mmlu.config import PruningConfig
from llm_pruning_mmlu.finetuning.config import MaskPolicyConfig
from llm_pruning_mmlu.finetuning.mask_policy import MaskEnforcer, MaskEnforcerCallback
from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.dispatch import prune_model
from llm_pruning_mmlu.pruning.semi_structured import compute_nm_magnitude_masks
from llm_pruning_mmlu.pruning.structured_targets import discover_mlp_channel_groups
from llm_pruning_mmlu.pruning.structured import compute_structured_masks
from llm_pruning_mmlu.pruning.targets import PruningParameter, find_pruning_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


class _LinearModel(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lm_head = nn.Linear(in_features, 16, bias=False)


def _policy(preserve: bool = True) -> MaskPolicyConfig:
    return MaskPolicyConfig(preserve_base_masks=preserve, mask_lora_pruned_channels=False)


def _semi_structured_enforcer(model: nn.Module, preserve: bool = True) -> MaskEnforcer:
    cfg = PruningConfig(
        method="global_magnitude_semi_structured",
        structure="nm_2_4",
        nm_n=2, nm_m=4, block_dim=1,
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    return MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy(preserve))


def _structured_enforcer(model: nn.Module, preserve: bool = True) -> MaskEnforcer:
    cfg = PruningConfig(
        method="global_magnitude_structured",
        structure="mlp_channel",
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    return MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy(preserve))


# ---------------------------------------------------------------------------
# pruning_kind classification
# ---------------------------------------------------------------------------

def test_from_prune_result_kind_structured():
    model = _MlpModel()
    enforcer = _structured_enforcer(model)
    assert enforcer.pruning_kind == "structured"


def test_from_prune_result_kind_semi_structured_magnitude():
    model = _LinearModel(in_features=8, out_features=4)
    enforcer = _semi_structured_enforcer(model)
    assert enforcer.pruning_kind == "semi_structured"


def test_from_prune_result_kind_semi_structured_wanda():
    model = _LinearModel(in_features=8, out_features=4)
    cfg = PruningConfig(
        method="wanda_semi_structured",
        structure="nm_2_4",
        nm_n=2, nm_m=4, block_dim=1,
        calibration_samples=4,
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    enforcer = MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy())
    assert enforcer.pruning_kind == "semi_structured"


def test_from_prune_result_kind_unstructured():
    model = _LinearModel(in_features=8, out_features=4)
    cfg = PruningConfig(
        method="global_magnitude_unstructured",
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    enforcer = MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy())
    assert enforcer.pruning_kind == "none"


# ---------------------------------------------------------------------------
# active property
# ---------------------------------------------------------------------------

def test_active_true_when_masks_present_and_preserve_true():
    model = _LinearModel(in_features=8, out_features=4)
    enforcer = _semi_structured_enforcer(model, preserve=True)
    assert enforcer.active is True


def test_active_false_when_preserve_false():
    model = _LinearModel(in_features=8, out_features=4)
    enforcer = _semi_structured_enforcer(model, preserve=False)
    assert enforcer.active is False


def test_active_false_when_masks_none():
    model = _LinearModel(in_features=8, out_features=4)
    cfg = PruningConfig(
        method="global_magnitude_unstructured",
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    # unstructured returns masks=None
    enforcer = MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy(True))
    assert enforcer.active is False


# ---------------------------------------------------------------------------
# enforce() — semi-structured
# ---------------------------------------------------------------------------

def test_enforce_re_zeros_pruned_positions_semi_structured():
    """After simulated optimizer step that regrows weights, enforce() must re-zero them."""
    model = _LinearModel(in_features=8, out_features=4)
    nn.init.normal_(model.fc.weight)
    enforcer = _semi_structured_enforcer(model, preserve=True)

    # Simulate optimizer step that partially regrows pruned weights
    with torch.no_grad():
        model.fc.weight.add_(torch.randn_like(model.fc.weight) * 0.01)

    enforcer.enforce()

    # After re-enforcement, N:M structure must hold
    from llm_pruning_mmlu.pruning.semi_structured import validate_nm_mask
    mask = enforcer.masks["fc.weight"]
    result = validate_nm_mask(model.fc.weight, mask, n=2, m=4)
    assert result["status"] == "ok"


def test_enforce_noop_when_preserve_false_semi_structured():
    """With preserve_base=False, enforce() must not change weights."""
    model = _LinearModel(in_features=8, out_features=4)
    nn.init.normal_(model.fc.weight)
    enforcer = _semi_structured_enforcer(model, preserve=False)

    with torch.no_grad():
        model.fc.weight.fill_(999.0)

    enforcer.enforce()
    # Weight must remain at 999 (enforce was a no-op)
    assert (model.fc.weight == 999.0).all()


def test_enforce_noop_when_masks_none():
    """Unstructured pruning (masks=None) → enforce() must be a no-op."""
    model = _LinearModel(in_features=8, out_features=4)
    cfg = PruningConfig(
        method="global_magnitude_unstructured",
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    enforcer = MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy(True))

    with torch.no_grad():
        for item in params:
            item.parameter.data.fill_(1.0)

    enforcer.enforce()
    # No change — masks is None
    for item in params:
        assert (item.parameter.data == 1.0).all()


# ---------------------------------------------------------------------------
# enforce() — structured
# ---------------------------------------------------------------------------

def test_enforce_re_zeros_pruned_positions_structured():
    """MaskEnforcer must restore structured channel zeros after an optimizer step."""
    model = _MlpModel()
    enforcer = _structured_enforcer(model, preserve=True)

    with torch.no_grad():
        for item in enforcer.parameters:
            item.parameter.data.add_(torch.randn_like(item.parameter) * 0.01)

    enforcer.enforce()

    # Gate/up/down consistency must still hold
    for layer_idx in range(2):
        mlp = model.layers[layer_idx]
        gate_w = mlp.gate_proj.weight
        up_w = mlp.up_proj.weight
        down_w = mlp.down_proj.weight
        for j in range(gate_w.shape[0]):
            gate_zero = gate_w[j].abs().sum() == 0
            up_zero = up_w[j].abs().sum() == 0
            down_zero = down_w[:, j].abs().sum() == 0
            assert gate_zero == up_zero == down_zero, (
                f"Layer {layer_idx} channel {j} inconsistent after enforce(): "
                f"gate_zero={gate_zero}, up_zero={up_zero}, down_zero={down_zero}"
            )


# ---------------------------------------------------------------------------
# MaskEnforcerCallback
# ---------------------------------------------------------------------------

def test_callback_calls_enforce_on_step_end():
    """MaskEnforcerCallback.on_step_end must call enforcer.enforce()."""
    enforcer = MagicMock(spec=MaskEnforcer)
    cb = MaskEnforcerCallback(enforcer)

    args = MagicMock()
    state = MagicMock()
    control = MagicMock()

    cb.on_step_end(args, state, control)
    enforcer.enforce.assert_called_once()


def test_callback_returns_control():
    """on_step_end must return the control object."""
    enforcer = MagicMock(spec=MaskEnforcer)
    cb = MaskEnforcerCallback(enforcer)

    args = MagicMock()
    state = MagicMock()
    control = MagicMock()

    result = cb.on_step_end(args, state, control)
    assert result is control


def test_callback_multiple_steps_enforce_called_each_time():
    enforcer = MagicMock(spec=MaskEnforcer)
    cb = MaskEnforcerCallback(enforcer)
    args, state, control = MagicMock(), MagicMock(), MagicMock()

    for _ in range(5):
        cb.on_step_end(args, state, control)

    assert enforcer.enforce.call_count == 5


# ---------------------------------------------------------------------------
# Wanda masks work with MaskEnforcer
# ---------------------------------------------------------------------------

def test_enforce_works_with_wanda_masks():
    """MaskEnforcer is pruning-kind-agnostic — Wanda masks should work identically."""
    model = _LinearModel(in_features=8, out_features=4)
    nn.init.normal_(model.fc.weight)
    cfg = PruningConfig(
        method="wanda_semi_structured",
        structure="nm_2_4",
        nm_n=2, nm_m=4, block_dim=1,
        calibration_samples=4,
        exclude_module_name_patterns=["lm_head"],
    )
    params, _, masks = prune_model(model, cfg, 50.0)
    enforcer = MaskEnforcer.from_prune_result(params, masks, cfg.method, _policy(True))
    assert enforcer.pruning_kind == "semi_structured"
    assert enforcer.active is True

    # Simulate regrowth then enforce
    with torch.no_grad():
        model.fc.weight.add_(torch.randn_like(model.fc.weight) * 0.01)

    enforcer.enforce()

    from llm_pruning_mmlu.pruning.semi_structured import validate_nm_mask
    result = validate_nm_mask(model.fc.weight, masks["fc.weight"], n=2, m=4)
    assert result["status"] == "ok"
