from __future__ import annotations

import torch

from llm_pruning_mmlu.pruning.targets import find_pruning_parameters


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
        self.lm_head = torch.nn.Linear(4, 2)


def test_finds_linear_weights_and_excludes_lm_head_bias():
    model = Tiny()
    params = find_pruning_parameters(model, exclude_module_name_patterns=["lm_head"])
    assert [item.name for item in params] == ["block.0.weight"]
    assert all(item.parameter_name == "weight" for item in params)
