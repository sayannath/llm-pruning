from __future__ import annotations

import torch


def resolve_torch_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype in (None, "auto"):
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if mapping[dtype] is torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return mapping[dtype]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
