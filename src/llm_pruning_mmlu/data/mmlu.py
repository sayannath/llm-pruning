from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from llm_pruning_mmlu.data.prompting import normalize_answer


def _from_fixture(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            row.setdefault("subject", "fixture")
            row.setdefault("question_id", f"fixture-{idx}")
            row["answer"] = normalize_answer(row["answer"])
            rows.append(row)
    return rows


def load_mmlu(
    hf_id: str = "cais/mmlu",
    split: str = "test",
    subjects: str | list[str] = "all",
    max_samples: int | None = None,
    fixture_path: str | None = None,
) -> list[dict[str, Any]]:
    if fixture_path:
        rows = _from_fixture(fixture_path)
    else:
        config_name = "all" if hf_id == "cais/mmlu" else None
        dataset = (
            load_dataset(hf_id, config_name, split=split)
            if config_name
            else load_dataset(hf_id, split=split)
        )
        rows = []
        allowed = None if subjects == "all" else set(subjects)
        for idx, row in enumerate(dataset):
            subject = row.get("subject", row.get("category", "unknown"))
            if allowed is not None and subject not in allowed:
                continue
            rows.append(
                {
                    "question_id": str(row.get("id", idx)),
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": normalize_answer(row["answer"]),
                }
            )
    rows.sort(key=lambda item: (item["subject"], item["question_id"]))
    return rows[:max_samples] if max_samples is not None else rows
