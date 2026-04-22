"""Tests for Wanda semi-structured N:M pruning (src/llm_pruning_mmlu/pruning/wanda.py).

Covers:
- Magnitude-fallback path (no model/tokenizer)
- N:M block correctness
- Activation-weighted path with a mock model
- Stats keys
- Remainder handling
- Dispatcher integration (wanda_semi_structured method)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.config import PruningConfig
from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.dispatch import prune_model
from llm_pruning_mmlu.pruning.semi_structured import validate_nm_mask
from llm_pruning_mmlu.pruning.targets import PruningParameter, find_pruning_parameters
from llm_pruning_mmlu.pruning.wanda import compute_nm_wanda_masks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(tensor: torch.Tensor, name: str = "weight", module_name: str = "fc") -> PruningParameter:
    p = nn.Parameter(tensor.clone())
    return PruningParameter(name=f"{module_name}.weight", module_name=module_name,
                            parameter_name="weight", parameter=p)


class _LinearModel(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lm_head = nn.Linear(in_features, 16, bias=False)


def _params_from_model(model: nn.Module) -> list[PruningParameter]:
    return find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])


_WANDA_24_CFG = PruningConfig(
    method="wanda_semi_structured",
    structure="nm_2_4",
    nm_n=2,
    nm_m=4,
    block_dim=1,
    calibration_samples=4,
    exclude_module_name_patterns=["lm_head"],
)


# ---------------------------------------------------------------------------
# Fallback path (no model/tokenizer) — behaves like magnitude N:M
# ---------------------------------------------------------------------------

def test_fallback_produces_correct_block_structure():
    """Without model/tokenizer, Wanda falls back to magnitude scoring."""
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    masks, _ = compute_nm_wanda_masks(params, n=2, m=4)
    mask = masks["fc.weight"]
    zeros_per_block = (~mask).reshape(4, 2, 4).sum(dim=2)
    assert (zeros_per_block == 2).all(), f"Expected 2 zeros per block, got: {zeros_per_block}"


def test_fallback_prunes_lowest_magnitude():
    """Fallback must zero the two smallest-magnitude weights per block."""
    row = torch.tensor([[1.0, 10.0, 2.0, 20.0]])
    params = [_make_param(row)]
    masks, _ = compute_nm_wanda_masks(params, n=2, m=4)
    mask = masks["fc.weight"][0]
    assert not mask[0], "magnitude 1.0 should be pruned"
    assert not mask[2], "magnitude 2.0 should be pruned"
    assert mask[1], "magnitude 10.0 should be kept"
    assert mask[3], "magnitude 20.0 should be kept"


def test_fallback_mask_shape_preserved():
    model = _LinearModel(in_features=8, out_features=4)
    params = _params_from_model(model)
    masks, _ = compute_nm_wanda_masks(params, n=2, m=4)
    for item in params:
        assert masks[item.name].shape == item.parameter.shape


def test_fallback_validate_nm_mask_passes():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    masks, _ = compute_nm_wanda_masks(params, n=2, m=4)
    result = validate_nm_mask(w, masks["fc.weight"], n=2, m=4)
    assert result["status"] == "ok"
    assert result["violations"] == 0


def test_fallback_apply_zeros_pruned_weights():
    model = _LinearModel(in_features=8, out_features=4)
    nn.init.normal_(model.fc.weight)
    params = _params_from_model(model)
    masks, _ = compute_nm_wanda_masks(params, n=2, m=4)
    apply_masks(params, masks)
    w = model.fc.weight
    zeros_per_block = (w == 0).reshape(4, 2, 4).sum(dim=2)
    assert (zeros_per_block == 2).all()


# ---------------------------------------------------------------------------
# 4:8 fallback
# ---------------------------------------------------------------------------

def test_fallback_4_8_correct_zeros_per_block():
    w = torch.randn(4, 16)
    params = [_make_param(w)]
    masks, _ = compute_nm_wanda_masks(params, n=4, m=8)
    mask = masks["fc.weight"]
    zeros_per_block = (~mask).reshape(4, 2, 8).sum(dim=2)
    assert (zeros_per_block == 4).all()


# ---------------------------------------------------------------------------
# Remainder handling
# ---------------------------------------------------------------------------

def test_remainder_weights_kept():
    """Cols 0-3 form one complete 2:4 block; cols 4-5 are remainder and must be kept."""
    w = torch.randn(2, 6)
    params = [_make_param(w)]
    masks, stats = compute_nm_wanda_masks(params, n=2, m=4)
    mask = masks["fc.weight"]
    zeros_complete = (~mask[:, :4]).sum(dim=1)
    assert (zeros_complete == 2).all()
    assert mask[:, 4:].all(), "remainder weights must be kept"
    assert stats["num_remainder_weights"] == 4


def test_in_features_less_than_m_nothing_pruned():
    w = torch.randn(4, 2)
    params = [_make_param(w)]
    masks, stats = compute_nm_wanda_masks(params, n=2, m=4)
    assert masks["fc.weight"].all(), "nothing should be pruned when in_features < m"
    assert stats["num_blocks_total"] == 0
    assert stats["nm_sparsity"] == 0.0


# ---------------------------------------------------------------------------
# Stats keys
# ---------------------------------------------------------------------------

def test_fallback_stats_keys_present():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    _, stats = compute_nm_wanda_masks(params, n=2, m=4)
    for key in (
        "nm_n", "nm_m", "block_dim",
        "num_blocks_total", "num_complete_blocks",
        "num_remainder_weights", "num_weights_pruned_by_nm", "nm_sparsity",
        "wanda_calibration_samples", "wanda_modules_with_norms",
    ):
        assert key in stats, f"Missing stats key: {key}"


def test_fallback_no_norms_collected():
    """Without model/tokenizer, wanda_modules_with_norms must be 0."""
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    _, stats = compute_nm_wanda_masks(params, n=2, m=4)
    assert stats["wanda_modules_with_norms"] == 0


# ---------------------------------------------------------------------------
# Activation-weighted path (mock model + tokenizer)
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer that tokenises any text to a fixed short sequence."""
    def __call__(self, text, max_length=512, truncation=True, return_tensors="pt"):
        ids = torch.randint(0, 1000, (1, min(max_length, 8)))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _ActivationModel(nn.Module):
    """Tiny causal-like model — just enough for forward hooks to fire."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)

    def forward(self, input_ids=None, **kwargs):
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 8)
        return self.fc(x)


def test_wanda_with_model_collects_norms():
    """With model/tokenizer, wanda_modules_with_norms should be > 0."""
    model = _ActivationModel()
    tokenizer = _MockTokenizer()
    w = torch.randn(4, 8)
    params = [
        PruningParameter(
            name="fc.weight",
            module_name="fc",
            parameter_name="weight",
            parameter=model.fc.weight,
        )
    ]
    _, stats = compute_nm_wanda_masks(
        params, n=2, m=4,
        model=model, tokenizer=tokenizer, calibration_samples=4,
    )
    assert stats["wanda_modules_with_norms"] > 0


def test_wanda_with_model_still_produces_valid_nm_mask():
    """Even with activation weighting, the output masks must satisfy the N:M constraint."""
    model = _ActivationModel()
    tokenizer = _MockTokenizer()
    params = [
        PruningParameter(
            name="fc.weight",
            module_name="fc",
            parameter_name="weight",
            parameter=model.fc.weight,
        )
    ]
    masks, _ = compute_nm_wanda_masks(
        params, n=2, m=4,
        model=model, tokenizer=tokenizer, calibration_samples=4,
    )
    result = validate_nm_mask(model.fc.weight, masks["fc.weight"], n=2, m=4)
    assert result["status"] == "ok"
    assert result["violations"] == 0


def test_wanda_with_uniform_activations_matches_magnitude():
    """If all activation norms are equal, Wanda score reduces to |W| → same result as magnitude."""
    torch.manual_seed(0)
    w = torch.randn(2, 8)
    param_mag = _make_param(w.clone())
    param_wanda = _make_param(w.clone(), module_name="fc2")

    # Magnitude masks
    from llm_pruning_mmlu.pruning.semi_structured import compute_nm_magnitude_masks
    mag_masks, _ = compute_nm_magnitude_masks([param_mag], n=2, m=4)

    # Wanda masks with model that produces uniform activations
    class _UniformActivationModel(nn.Module):
        def __init__(self, fc_param):
            super().__init__()
            self.fc2 = nn.Linear(8, 2, bias=False)
            self.fc2.weight = fc_param

        def forward(self, input_ids=None, **kwargs):
            # Uniform input: every column has identical activation
            x = torch.ones(1, 1, 8)
            return self.fc2(x)

    fc2_param = nn.Parameter(w.clone())
    model = _UniformActivationModel(fc2_param)
    tokenizer = _MockTokenizer()
    pruning_param = PruningParameter(
        name="fc2.weight", module_name="fc2", parameter_name="weight",
        parameter=model.fc2.weight,
    )
    wanda_masks, _ = compute_nm_wanda_masks(
        [pruning_param], n=2, m=4,
        model=model, tokenizer=tokenizer, calibration_samples=8,
    )

    # Both should prune the same positions since uniform activation ∝ magnitude
    assert torch.equal(mag_masks["fc.weight"], wanda_masks["fc2.weight"])


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------

class _SmallLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)


def test_dispatch_wanda_zero_sparsity_no_change():
    model = _SmallLinearModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prune_model(model, _WANDA_24_CFG, 0.0)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"{n} changed at sparsity=0"


def test_dispatch_wanda_stats_keys():
    model = _SmallLinearModel()
    _, stats, masks = prune_model(model, _WANDA_24_CFG, 50.0)
    for key in (
        "total", "nonzero", "sparsity",
        "nm_n", "nm_m", "nm_sparsity",
        "method", "structure",
        "wanda_calibration_samples",
    ):
        assert key in stats, f"Missing stats key: {key}"
    assert stats["method"] == "wanda_semi_structured"
    assert stats["nm_n"] == 2
    assert stats["nm_m"] == 4


def test_dispatch_wanda_achieves_50_percent_nm_sparsity():
    model = _SmallLinearModel()
    _, stats, _ = prune_model(model, _WANDA_24_CFG, 50.0)
    # fc1 and fc2 both have in_features=8 (divisible by 4) → no remainder
    assert abs(stats["nm_sparsity"] - 50.0) < 1e-4


def test_dispatch_wanda_returns_masks():
    model = _SmallLinearModel()
    _, _, masks = prune_model(model, _WANDA_24_CFG, 50.0)
    assert masks is not None
    assert len(masks) > 0


def test_dispatch_wanda_masks_pass_nm_validation():
    model = _SmallLinearModel()
    _, _, masks = prune_model(model, _WANDA_24_CFG, 50.0)
    assert masks is not None
    for name, mask in masks.items():
        param = dict(model.named_parameters())[name]
        result = validate_nm_mask(param, mask, n=2, m=4)
        assert result["status"] == "ok", f"{name}: {result}"


def test_dispatch_wanda_lm_head_excluded():
    model = _SmallLinearModel()
    lm_head_before = model.lm_head.weight.detach().clone()
    prune_model(model, _WANDA_24_CFG, 50.0)
    assert torch.equal(model.lm_head.weight, lm_head_before)


def test_dispatch_wanda_no_group_stats():
    model = _SmallLinearModel()
    _, stats, _ = prune_model(model, _WANDA_24_CFG, 50.0)
    assert "num_groups_total" not in stats


# ---------------------------------------------------------------------------
# Guard: invalid pattern
# ---------------------------------------------------------------------------

def test_invalid_pattern_raises():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    try:
        compute_nm_wanda_masks(params, n=1, m=3)
        assert False, "Expected ValueError for unsupported pattern"
    except ValueError:
        pass


def test_invalid_block_dim_raises():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    try:
        compute_nm_wanda_masks(params, n=2, m=4, block_dim=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
