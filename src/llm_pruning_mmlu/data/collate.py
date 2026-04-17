from __future__ import annotations


def batch_iter(items: list[dict], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
