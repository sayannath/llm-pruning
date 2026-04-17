from __future__ import annotations

from llm_pruning_mmlu.config import load_config, load_config_dict


def test_config_inheritance_and_list_replacement():
    data = load_config_dict("tests/fixtures/dummy_config.yaml")
    assert data["seed"] == 42
    assert data["pruning"]["sparsities"] == [0, 50]
    assert data["dataset"]["fixture_path"] == "tests/fixtures/tiny_mmlu.jsonl"


def test_parse_config():
    cfg = load_config("tests/fixtures/dummy_config.yaml")
    assert cfg.model.name == "dummy"
    assert cfg.device.dtype == "bfloat16"
