from __future__ import annotations

import torch

from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.magnitude import compute_global_magnitude_masks
from llm_pruning_mmlu.pruning.targets import find_pruning_parameters


def test_global_pruning_exact_count():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(3, 1, bias=False)
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]))
        model[1].weight.copy_(torch.tensor([[7.0, -8.0, 9.0]]))
    params = find_pruning_parameters(model)
    masks = compute_global_magnitude_masks(params, 50)
    apply_masks(params, masks)
    total = sum(item.parameter.numel() for item in params)
    zeros = sum(int((item.parameter == 0).sum().item()) for item in params)
    assert total == 9
    assert zeros == 4


def test_zero_sparsity_keeps_weights():
    model = torch.nn.Linear(2, 2, bias=False)
    before = model.weight.detach().clone()
    params = find_pruning_parameters(model)
    apply_masks(params, compute_global_magnitude_masks(params, 0))
    assert torch.equal(model.weight, before)
