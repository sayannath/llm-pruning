from __future__ import annotations

from llm_pruning_mmlu.utils.io import read_json, read_jsonl, write_csv, write_json, write_jsonl


def test_json_jsonl_csv_writers(tmp_path):
    write_json(tmp_path / "a" / "data.json", {"x": 1})
    assert read_json(tmp_path / "a" / "data.json") == {"x": 1}
    write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}, {"a": 2}])
    assert read_jsonl(tmp_path / "rows.jsonl") == [{"a": 1}, {"a": 2}]
    write_csv(tmp_path / "rows.csv", [{"b": 2, "a": 1}])
    assert (tmp_path / "rows.csv").read_text(encoding="utf-8").splitlines()[0] == "a,b"
