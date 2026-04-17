from __future__ import annotations

from typing import Any

import torch

from llm_pruning_mmlu.pruning.targets import PruningParameter


@torch.no_grad()
def pruning_stats(parameters: list[PruningParameter]) -> dict[str, Any]:
    total = 0
    nonzero = 0
    layers = []
    for item in parameters:
        tensor = item.parameter.detach()
        layer_total = tensor.numel()
        layer_nonzero = int(torch.count_nonzero(tensor).item())
        total += layer_total
        nonzero += layer_nonzero
        layers.append(
            {
                "name": item.name,
                "total": layer_total,
                "nonzero": layer_nonzero,
                "sparsity": 100.0 * (1.0 - layer_nonzero / layer_total) if layer_total else 0.0,
            }
        )
    return {
        "total": total,
        "nonzero": nonzero,
        "zero": total - nonzero,
        "sparsity": 100.0 * (1.0 - nonzero / total) if total else 0.0,
        "layers": layers,
    }
