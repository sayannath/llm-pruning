from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSlice:
    parameter_name: str
    parameter: torch.nn.Parameter
    dim: int
    index: int


@dataclass(frozen=True)
class StructuredGroup:
    name: str
    layer_name: str
    structure: str
    slices: tuple[TensorSlice, ...]


def _unwrap_linear(
    parent: torch.nn.Module, attr: str
) -> tuple[torch.nn.Linear | None, str]:
    """Return the Linear at parent.attr and its path suffix.

    Handles models like Gemma4 where projections are wrapped in a custom class
    (e.g. Gemma4ClippableLinear) that holds a plain Linear at .linear.  The
    returned suffix is used to build parameter names that match what
    find_pruning_parameters produces, e.g. "gate_proj.linear" instead of
    "gate_proj" for wrapped models.
    """
    module = getattr(parent, attr, None)
    if isinstance(module, torch.nn.Linear):
        return module, attr
    inner = getattr(module, "linear", None)
    if isinstance(inner, torch.nn.Linear):
        return inner, f"{attr}.linear"
    return None, attr


def discover_mlp_channel_groups(model: torch.nn.Module) -> list[StructuredGroup]:
    """Discover SwiGLU MLP channel groups across all transformer layers.

    Each group represents one intermediate channel j, spanning:
      gate_proj.weight[j, :]  (dim=0, row j)
      up_proj.weight[j, :]    (dim=0, row j)
      down_proj.weight[:, j]  (dim=1, col j)

    Groups are ordered layer-first, then channel index, enabling efficient
    batched scoring by iterating contiguous layer slices.
    """
    groups: list[StructuredGroup] = []

    for module_name, module in model.named_modules():
        gate, gate_suffix = _unwrap_linear(module, "gate_proj")
        up, up_suffix = _unwrap_linear(module, "up_proj")
        down, down_suffix = _unwrap_linear(module, "down_proj")
        if not (gate and up and down):
            continue

        intermediate_size = gate.weight.shape[0]
        if up.weight.shape[0] != intermediate_size:
            raise ValueError(
                f"{module_name}: up_proj rows {up.weight.shape[0]} != "
                f"gate_proj rows {intermediate_size}"
            )
        if down.weight.shape[1] != intermediate_size:
            raise ValueError(
                f"{module_name}: down_proj cols {down.weight.shape[1]} != "
                f"intermediate_size {intermediate_size}"
            )

        gate_name = f"{module_name}.{gate_suffix}.weight"
        up_name = f"{module_name}.{up_suffix}.weight"
        down_name = f"{module_name}.{down_suffix}.weight"

        for j in range(intermediate_size):
            groups.append(
                StructuredGroup(
                    name=f"{module_name}.ch{j}",
                    layer_name=module_name,
                    structure="mlp_channel",
                    slices=(
                        TensorSlice(gate_name, gate.weight, 0, j),
                        TensorSlice(up_name, up.weight, 0, j),
                        TensorSlice(down_name, down.weight, 1, j),
                    ),
                )
            )

    return groups
