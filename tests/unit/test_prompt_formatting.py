from __future__ import annotations

from llm_pruning_mmlu.data.prompting import format_mmlu_prompt, normalize_answer


def test_prompt_fixed_format():
    prompt = format_mmlu_prompt("Q?", ["a", "b", "c", "d"])
    assert prompt == "Question: Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:"


def test_normalize_answer():
    assert normalize_answer(2) == "C"
    assert normalize_answer("D") == "D"
