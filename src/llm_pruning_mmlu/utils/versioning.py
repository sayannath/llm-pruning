from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

import torch


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def runtime_metadata() -> dict[str, Any]:
    try:
        import datasets
        import transformers
    except Exception:
        datasets = None
        transformers = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": getattr(transformers, "__version__", None),
        "datasets": getattr(datasets, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": git_commit(),
    }
