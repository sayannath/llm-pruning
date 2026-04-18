from __future__ import annotations

import torch
import torch.nn as nn

from llm_pruning_mmlu.pruning.semi_structured import compute_nm_magnitude_masks
from llm_pruning_mmlu.pruning.targets import PruningParameter, find_pruning_parameters


def _make_param(tensor: torch.Tensor, name: str = "weight") -> PruningParameter:
    p = nn.Parameter(tensor.clone())
    return PruningParameter(name=name, module_name="fc", parameter_name="weight", parameter=p)


# ---------------------------------------------------------------------------
# Stats keys present
# ---------------------------------------------------------------------------

def test_stats_keys_present():
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    for key in (
        "nm_n", "nm_m", "block_dim",
        "num_blocks_total", "num_complete_blocks",
        "num_remainder_weights", "num_weights_pruned_by_nm", "nm_sparsity",
    ):
        assert key in stats, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Block counts
# ---------------------------------------------------------------------------

def test_2_4_block_counts_exact():
    """2 rows × 4 cols with m=4: 2 complete blocks, no remainder."""
    w = torch.randn(2, 4)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    assert stats["num_blocks_total"] == 2      # 2 rows × 1 block/row
    assert stats["num_complete_blocks"] == 2
    assert stats["num_remainder_weights"] == 0
    assert stats["num_weights_pruned_by_nm"] == 4  # 2 blocks × 2 pruned each


def test_2_4_block_counts_with_remainder():
    """2 rows × 6 cols, m=4: 2 complete blocks + 4 remainder weights."""
    w = torch.randn(2, 6)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    assert stats["num_blocks_total"] == 2
    assert stats["num_remainder_weights"] == 4   # 2 rows × 2 remainder cols
    assert stats["num_weights_pruned_by_nm"] == 4


def test_4_8_block_counts():
    """3 rows × 16 cols, m=8: 6 complete blocks, 0 remainder."""
    w = torch.randn(3, 16)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=4, m=8)
    assert stats["num_blocks_total"] == 6
    assert stats["num_remainder_weights"] == 0
    assert stats["num_weights_pruned_by_nm"] == 24  # 6 blocks × 4 pruned each


# ---------------------------------------------------------------------------
# nm_sparsity
# ---------------------------------------------------------------------------

def test_nm_sparsity_is_50_for_complete_blocks():
    """When every weight is in a complete block, nm_sparsity must be exactly 50%."""
    w = torch.randn(4, 8)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    assert abs(stats["nm_sparsity"] - 50.0) < 1e-6


def test_nm_sparsity_zero_when_no_complete_blocks():
    """in_features < m: no complete blocks, nm_sparsity must be 0."""
    w = torch.randn(4, 2)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    assert stats["nm_sparsity"] == 0.0


# ---------------------------------------------------------------------------
# Multi-layer aggregation
# ---------------------------------------------------------------------------

def test_stats_aggregate_across_multiple_parameters():
    """Stats should sum across all parameters."""
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4, bias=False)
            self.fc2 = nn.Linear(8, 4, bias=False)
    model = _Model()
    params = find_pruning_parameters(model)
    _, stats = compute_nm_magnitude_masks(params, n=2, m=4)
    # fc1: 4 rows × 2 blocks/row = 8 blocks; fc2: same → 16 blocks total
    assert stats["num_blocks_total"] == 16
    assert stats["num_weights_pruned_by_nm"] == 32  # 16 blocks × 2 pruned each


# ---------------------------------------------------------------------------
# nm_n, nm_m, block_dim are echoed back
# ---------------------------------------------------------------------------

def test_stats_echo_nm_params():
    w = torch.randn(2, 8)
    params = [_make_param(w)]
    _, stats = compute_nm_magnitude_masks(params, n=4, m=8)
    assert stats["nm_n"] == 4
    assert stats["nm_m"] == 8
    assert stats["block_dim"] == 1
