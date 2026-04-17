from __future__ import annotations

CHOICES = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    if len(choices) != 4:
        raise ValueError(f"MMLU examples must have four choices, got {len(choices)}")
    lines = [f"Question: {question}"]
    lines.extend(f"{label}. {choice}" for label, choice in zip(CHOICES, choices, strict=True))
    lines.append("Answer:")
    return "\n".join(lines)


def normalize_answer(answer: int | str) -> str:
    if isinstance(answer, int):
        return CHOICES[answer]
    answer = str(answer).strip()
    if answer in CHOICES:
        return answer
    if answer.isdigit():
        return CHOICES[int(answer)]
    raise ValueError(f"Unsupported answer label: {answer}")
