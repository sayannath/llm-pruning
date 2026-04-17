from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(data: Any, length: int = 12) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
