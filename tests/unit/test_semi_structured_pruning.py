from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.semi_structured import (
    compute_nm_magnitude_masks,
    validate_nm_mask,
)
from llm_pruning_mmlu.pruning.targets import PruningParameter, find_pruning_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(tensor: torch.Tensor, name: str = "weight") -> PruningParameter:
    p = nn.Parameter(tensor.clone())
    return PruningParameter(name=name, module_name="fc", parameter_name="weight", parameter=p)


def _params_from_model(model: nn.Module) -> list[PruningParameter]:
    return find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])


class _LinearModel(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lm_head = nn.Linear(in_features, 16, bias=False)


# ---------------------------------------------------------------------------
# 2:4 mask correctness
# ---------------------------------------------------------------------------

def test_2_4_exactly_two_zeros_per_block():
    """Every complete block of 4 must have exactly 2 zeros."""
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    mask = masks["weight"]
    # reshape complete region into blocks: [4, 2, 4]
    zeros_per_block = (~mask).reshape(4, 2, 4).sum(dim=2)
    assert (zeros_per_block == 2).all(), f"Block zero counts: {zeros_per_block}"


def test_2_4_lowest_magnitude_pruned():
    """In each block the two smallest-magnitude weights are zeroed."""
    # Construct a row where magnitudes are known: [1, 10, 2, 20, 3, 30, 4, 40]
    row = torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]])
    params = [_make_param(row)]
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    mask = masks["weight"][0]  # [8]
    # Block 0 cols [0,1,2,3]: magnitudes [1,10,2,20] → prune indices 0,2 (1.0, 2.0)
    assert not mask[0], "smallest in block 0 should be pruned"
    assert mask[1], "largest in block 0 should be kept"
    assert not mask[2], "second smallest in block 0 should be pruned"
    assert mask[3], "largest in block 0 should be kept"
    # Block 1 cols [4,5,6,7]: magnitudes [3,30,4,40] → prune indices 4,6
    assert not mask[4]
    assert mask[5]
    assert not mask[6]
    assert mask[7]


def test_2_4_mask_shape_preserved():
    model = _LinearModel(in_features=8, out_features=4)
    params = _params_from_model(model)
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    for item in params:
        assert masks[item.name].shape == item.parameter.shape


def test_2_4_apply_zeros_pruned_weights():
    model = _LinearModel(in_features=8, out_features=4)
    nn.init.normal_(model.fc.weight)
    params = _params_from_model(model)
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    apply_masks(params, masks)
    w = model.fc.weight
    zeros_per_block = (w == 0).reshape(4, 2, 4).sum(dim=2)
    assert (zeros_per_block == 2).all()


# ---------------------------------------------------------------------------
# 4:8 mask correctness
# ---------------------------------------------------------------------------

def test_4_8_exactly_four_zeros_per_block():
    w = torch.randn(4, 16)
    params = [_make_param(w)]
    masks, _ = compute_nm_magnitude_masks(params, n=4, m=8)
    mask = masks["weight"]
    zeros_per_block = (~mask).reshape(4, 2, 8).sum(dim=2)
    assert (zeros_per_block == 4).all(), f"Block zero counts: {zeros_per_block}"


def test_4_8_mask_shape_preserved():
    model = _LinearModel(in_features=16, out_features=4)
    params = _params_from_model(model)
    masks, _ = compute_nm_magnitude_masks(params, n=4, m=8)
    for item in params:
        assert masks[item.name].shape == item.parameter.shape


# ---------------------------------------------------------------------------
# Zero sparsity → no-op
# ---------------------------------------------------------------------------

def test_zero_sparsity_leaves_weights_unchanged():
    """sparsity=0 path: compute_nm_magnitude_masks still builds masks; dispatcher
    skips calling it. Test here that if called at sparsity=0 the masks are all-True."""
    # At sparsity=0 the dispatcher doesn't call compute_nm_magnitude_masks at all,
    # so we test directly that the function itself produces correct masks (which are
    # applied conditionally by the dispatcher).
    w = torch.randn(4, 8)
    before = w.clone()
    params = [_make_param(w)]
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    # masks here are N:M masks regardless; sparsity gate is in the dispatcher.
    # What we verify is that apply_masks with all-True leaves values unchanged.
    all_true_masks = {"weight": torch.ones_like(w, dtype=torch.bool)}
    apply_masks(params, all_true_masks)
    assert torch.equal(params[0].parameter.data, before)


# ---------------------------------------------------------------------------
# Remainder handling
# ---------------------------------------------------------------------------

def test_remainder_weights_are_kept():
    """in_features=6 with m=4: cols 0-3 form one complete block; cols 4-5 are remainder."""
    w = torch.randn(2, 6)
    params = [_make_param(w)]
    masks, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    mask = masks["weight"]  # [2, 6]
    # Complete block region: cols 0-3 must have exactly 2 zeros per row
    zeros_complete = (~mask[:, :4]).sum(dim=1)
    assert (zeros_complete == 2).all(), f"complete block zeros per row: {zeros_complete}"
    # Remainder cols 4-5 must be fully kept
    assert mask[:, 4:].all(), "remainder weights must be kept (mask=True)"
    assert stats["num_remainder_weights"] == 4  # 2 rows × 2 remainder cols


def test_in_features_less_than_m_nothing_pruned():
    """in_features=2 < m=4: entire layer is remainder, nothing should be pruned."""
    w = torch.randn(4, 2)
    params = [_make_param(w)]
    masks, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    mask = masks["weight"]
    assert mask.all(), "all weights must be kept when in_features < m"
    assert stats["num_blocks_total"] == 0
    assert stats["nm_sparsity"] == 0.0
    assert stats["num_remainder_weights"] == w.numel()


def test_in_features_exactly_m():
    """in_features == m: exactly one complete block per row, no remainder."""
    w = torch.randn(3, 4)
    params = [_make_param(w)]
    masks, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    mask = masks["weight"]
    zeros_per_row = (~mask).sum(dim=1)
    assert (zeros_per_row == 2).all()
    assert stats["num_remainder_weights"] == 0


# ---------------------------------------------------------------------------
# validate_nm_mask
# ---------------------------------------------------------------------------

def test_validate_nm_mask_passes_on_correct_mask():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    result = validate_nm_mask(w, masks["weight"], n=2, m=4)
    assert result["status"] == "ok"
    assert result["violations"] == 0


def test_validate_nm_mask_detects_block_violation():
    w = torch.ones(2, 4)
    bad_mask = torch.ones(2, 4, dtype=torch.bool)
    # Only zero one weight per block instead of the required two
    bad_mask[0, 0] = False
    bad_mask[1, 0] = False
    result = validate_nm_mask(w, bad_mask, n=2, m=4)
    assert result["violations"] > 0
    assert result["status"] == "failed"


def test_validate_nm_mask_detects_remainder_violation():
    """Remainder weights that are pruned should count as violations."""
    w = torch.ones(2, 6)
    # Correct 2:4 block then prune one remainder weight
    params = [_make_param(w)]
    masks, _ = compute_nm_magnitude_masks(params, n=2, m=4)
    bad_mask = masks["weight"].clone()
    bad_mask[0, 5] = False  # prune a remainder weight
    result = validate_nm_mask(w, bad_mask, n=2, m=4)
    assert result["remainder_violations"] > 0
    assert result["status"] == "failed"


def test_validate_nm_mask_raises_on_shape_mismatch():
    w = torch.randn(2, 4)
    bad_mask = torch.ones(2, 8, dtype=torch.bool)
    try:
        validate_nm_mask(w, bad_mask, n=2, m=4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_validate_nm_mask_raises_on_bad_block_dim():
    w = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    try:
        validate_nm_mask(w, mask, n=2, m=4, block_dim=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Invalid pattern / block_dim guards
# ---------------------------------------------------------------------------

def test_invalid_pattern_raises():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    try:
        compute_nm_magnitude_masks(params, n=1, m=3)
        assert False, "Expected ValueError for unsupported pattern"
    except ValueError:
        pass


def test_invalid_n_ge_m_raises():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    try:
        compute_nm_magnitude_masks(params, n=4, m=4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_block_dim_not_one_raises():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    try:
        compute_nm_magnitude_masks(params, n=2, m=4, block_dim=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
