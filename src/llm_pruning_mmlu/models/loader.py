from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_pruning_mmlu.config import DeviceConfig, ModelConfig
from llm_pruning_mmlu.utils.device import default_device, resolve_torch_dtype


class DummyTokenizer:
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, " A": 2, " B": 3, " C": 4, " D": 5}

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if text in self.vocab:
            ids = [self.vocab[text]]
        else:
            ids = [1]
            ids.extend(6 + (ord(char) % 32) for char in str(text)[-8:])
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor}
        return {"input_ids": ids}


class DummyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.lm_head.out_features, device=input_ids.device)
        logits[..., 2] = 0.1
        logits[..., 3] = 0.2
        logits[..., 4] = 0.3
        logits[..., 5] = 0.0
        return SimpleNamespace(logits=logits)


class _DummySwiGluMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        import torch.nn.functional as F
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DummyTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = _DummySwiGluMlp(hidden_size, intermediate_size)

    def forward(self, x):
        return self.mlp(x)


class DummyMlpCausalLM(torch.nn.Module):
    """Tiny transformer-like model with SwiGLU MLP blocks for structured pruning tests."""

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 8,
        intermediate_size: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_DummyTransformerBlock(hidden_size, intermediate_size) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.lm_head.out_features, device=input_ids.device)
        logits[..., 2] = 0.1
        logits[..., 3] = 0.2
        logits[..., 4] = 0.3
        logits[..., 5] = 0.0
        return SimpleNamespace(logits=logits)


def load_model_and_tokenizer(model_cfg: ModelConfig, device_cfg: DeviceConfig):
    if model_cfg.hf_id == "dummy/local":
        model = DummyCausalLM()
        model.eval()
        return model, DummyTokenizer()

    if model_cfg.hf_id == "dummy/mlp":
        model = DummyMlpCausalLM()
        model.eval()
        return model, DummyTokenizer()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dtype = resolve_torch_dtype(device_cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id,
        token=token,
        trust_remote_code=device_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": device_cfg.trust_remote_code,
        "token": token,
    }
    if device_cfg.load_in_4bit:
        kwargs["load_in_4bit"] = True
    if device_cfg.device_map:
        kwargs["device_map"] = device_cfg.device_map
    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, **kwargs)
    if not device_cfg.device_map:
        model.to(default_device())
    model.eval()
    return model, tokenizer
