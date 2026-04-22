"""Wanda-style N:M semi-structured pruning.

Score: abs(W[row, col]) * sqrt(mean_calib(X[:, col]^2))

Calibration inputs are fetched from MMLU auxiliary_train and run through the
model with forward hooks that collect the squared L2 norm of each Linear
layer's input activations.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

from llm_pruning_mmlu.pruning.semi_structured import _check_block_dim, _SUPPORTED_PATTERNS
from llm_pruning_mmlu.pruning.targets import PruningParameter

_log = logging.getLogger("llm_pruning_mmlu")

_CALIB_HF_ID = "cais/mmlu"
_CALIB_SPLIT = "auxiliary_train"
_CALIB_MAX_SEQ = 512


def _load_calibration_tokens(
    tokenizer,
    num_samples: int,
) -> list[torch.Tensor]:
    """Return a list of 1-D token-id tensors for calibration."""
    from datasets import load_dataset  # lazy import

    ds = load_dataset(_CALIB_HF_ID, "all", split=_CALIB_SPLIT, trust_remote_code=True)
    texts: list[str] = []
    for row in ds:
        texts.append(row["question"])
        if len(texts) >= num_samples:
            break

    token_lists: list[torch.Tensor] = []
    for text in texts:
        enc = tokenizer(
            text,
            max_length=_CALIB_MAX_SEQ,
            truncation=True,
            return_tensors="pt",
        )
        token_lists.append(enc["input_ids"].squeeze(0))
    return token_lists


@contextmanager
def _activation_hook_context(
    model: torch.nn.Module,
    module_names: set[str],
    sq_norms: dict[str, torch.Tensor],
    counts: dict[str, int],
):
    """Context manager that registers/removes forward pre-hooks collecting input sq norms."""
    handles = []

    def _make_hook(name: str):
        def _hook(module, args):
            if not args:
                return
            x = args[0].detach().float()
            # x shape: [batch, seq, in_features] or [batch, in_features]
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim == 1:
                x = x.unsqueeze(0)
            # accumulate sum of squared values per input feature
            col_sq = x.pow(2).sum(dim=0).cpu()
            n_tokens = x.shape[0]
            if name in sq_norms:
                sq_norms[name] = sq_norms[name] + col_sq
                counts[name] += n_tokens
            else:
                sq_norms[name] = col_sq
                counts[name] = n_tokens

        return _hook

    for mod_name, module in model.named_modules():
        if mod_name in module_names:
            h = module.register_forward_pre_hook(_make_hook(mod_name))
            handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def compute_nm_wanda_masks(
    parameters: list[PruningParameter],
    n: int,
    m: int,
    block_dim: int = 1,
    model: torch.nn.Module | None = None,
    tokenizer=None,
    calibration_samples: int = 128,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build N:M boolean masks using Wanda activation-weighted scoring.

    If model/tokenizer are not provided, falls back to magnitude scoring.
    """
    _check_block_dim(block_dim)
    if not (0 < n < m):
        raise ValueError(f"N:M pattern requires 0 < n < m; got n={n}, m={m}")
    if (n, m) not in _SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unsupported N:M pattern ({n}:{m}). Supported: {sorted(_SUPPORTED_PATTERNS)}"
        )

    sq_norms: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    if model is not None and tokenizer is not None:
        module_names = {item.module_name for item in parameters}
        _log.info(
            "Wanda: collecting activation norms from %d calibration samples ...",
            calibration_samples,
        )
        calib_tokens = _load_calibration_tokens(tokenizer, calibration_samples)
        device = next(model.parameters()).device
        with _activation_hook_context(model, module_names, sq_norms, counts):
            for ids in calib_tokens:
                input_ids = ids.unsqueeze(0).to(device)
                model(input_ids=input_ids)
        _log.info("Wanda: collected norms for %d modules", len(sq_norms))
    else:
        _log.warning(
            "Wanda: model/tokenizer not provided — falling back to magnitude scoring"
        )

    masks: dict[str, torch.Tensor] = {}
    total_blocks = 0
    total_remainder_weights = 0
    total_pruned = 0

    for item in parameters:
        w = item.parameter

        if w.ndim != 2:
            masks[item.name] = torch.ones_like(w, dtype=torch.bool)
            continue

        out_features, in_features = w.shape
        num_blocks_per_row = in_features // m
        remainder = in_features % m

        if num_blocks_per_row == 0:
            masks[item.name] = torch.ones_like(w, dtype=torch.bool)
            total_remainder_weights += w.numel()
            continue

        w_cpu = w.detach().to(device="cpu", dtype=torch.float32)

        # Compute per-column activation norm if available
        mod_name = item.module_name
        if mod_name in sq_norms and counts.get(mod_name, 0) > 0:
            act_norm = (sq_norms[mod_name] / counts[mod_name]).sqrt()  # [in_features]
            # Wanda score: abs(W) * activation_norm (broadcast over out_features)
            score = w_cpu.abs() * act_norm.unsqueeze(0)
        else:
            score = w_cpu.abs()

        complete_end = num_blocks_per_row * m
        # [out, num_blocks, m]
        score_blocks = score[:, :complete_end].reshape(out_features, num_blocks_per_row, m)
        order = score_blocks.argsort(dim=2, stable=True)  # ascending: lowest score first

        prune_local = torch.zeros(out_features, num_blocks_per_row, m, dtype=torch.bool)
        prune_local.scatter_(2, order[:, :, :n], True)

        mask = torch.ones(out_features, in_features, dtype=torch.bool)
        mask[:, :complete_end] = (~prune_local).reshape(out_features, complete_end)
        masks[item.name] = mask.to(device=w.device)

        layer_blocks = out_features * num_blocks_per_row
        total_blocks += layer_blocks
        total_remainder_weights += out_features * remainder
        total_pruned += layer_blocks * n

    total_weights_in_complete_blocks = total_blocks * m
    nm_sparsity = (
        100.0 * total_pruned / total_weights_in_complete_blocks
        if total_weights_in_complete_blocks > 0
        else 0.0
    )

    stats: dict[str, Any] = {
        "nm_n": n,
        "nm_m": m,
        "block_dim": block_dim,
        "num_blocks_total": total_blocks,
        "num_complete_blocks": total_blocks,
        "num_remainder_weights": total_remainder_weights,
        "num_weights_pruned_by_nm": total_pruned,
        "nm_sparsity": nm_sparsity,
        "wanda_calibration_samples": calibration_samples,
        "wanda_modules_with_norms": len(sq_norms),
    }
    return masks, stats
