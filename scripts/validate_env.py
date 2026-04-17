#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import torch


def main() -> None:
    try:
        import datasets
        import transformers
    except Exception as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"datasets={datasets.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(
        f"hf_token_present={bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}"
    )


if __name__ == "__main__":
    main()
