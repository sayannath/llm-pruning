from __future__ import annotations


def model_num_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
