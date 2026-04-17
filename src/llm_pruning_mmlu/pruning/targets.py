from __future__ import annotations

from dataclasses import dataclass
import fnmatch

import torch


@dataclass(frozen=True)
class PruningParameter:
    name: str
    module_name: str
    parameter_name: str
    parameter: torch.nn.Parameter


_MODULE_TYPES = {"Linear": torch.nn.Linear}


def _excluded(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) or pattern in name for pattern in patterns)


def find_pruning_parameters(
    model: torch.nn.Module,
    target_module_types: list[str] | None = None,
    target_parameter_names: list[str] | None = None,
    exclude_module_name_patterns: list[str] | None = None,
    prune_bias: bool = False,
) -> list[PruningParameter]:
    target_module_types = target_module_types or ["Linear"]
    target_parameter_names = target_parameter_names or ["weight"]
    exclude_module_name_patterns = exclude_module_name_patterns or []
    allowed_types = tuple(
        _MODULE_TYPES[name] for name in target_module_types if name in _MODULE_TYPES
    )
    if not allowed_types:
        raise ValueError(f"No supported target module types in {target_module_types}")
    params: list[PruningParameter] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, allowed_types) or _excluded(
            module_name, exclude_module_name_patterns
        ):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name == "bias" and not prune_bias:
                continue
            if parameter_name not in target_parameter_names:
                continue
            params.append(
                PruningParameter(
                    name=f"{module_name}.{parameter_name}" if module_name else parameter_name,
                    module_name=module_name,
                    parameter_name=parameter_name,
                    parameter=parameter,
                )
            )
    return params
