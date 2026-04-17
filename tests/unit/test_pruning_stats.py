from __future__ import annotations

import torch

from llm_pruning_mmlu.pruning.stats import pruning_stats
from llm_pruning_mmlu.pruning.targets import find_pruning_parameters


def test_stats_counts_zeros():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
    stats = pruning_stats(find_pruning_parameters(model))
    assert stats["total"] == 4
    assert stats["nonzero"] == 2
    assert stats["sparsity"] == 50.0
