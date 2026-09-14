"""Fast internal utilities for pismith: ids, fast JSON, env, time."""
from __future__ import annotations

import json
import os
import random
import time
import uuid

try:  # orjson is ~5-10x faster when available, stdlib otherwise
    import orjson as _orjson

    def fast_dumps(obj) -> str:
        return _orjson.dumps(obj, default=str).decode("utf-8")

    def fast_loads(s: str):
        return _orjson.loads(s)

    FAST_JSON = "orjson"
except Exception:
    def fast_dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    def fast_loads(s: str):
        return json.loads(s)

    FAST_JSON = "json"


def new_id(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


def utcnow() -> float:
    return time.time()


def sample_hit(rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def getenv(*names: str, default: str = "") -> str:
    """First non-empty env var wins.

    pismith reads ``PISMITH_*`` first and falls back to legacy ``PYSMITH_*``,
    so renaming the library never breaks existing deployments.
    """
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


__all__ = ["fast_dumps", "fast_loads", "FAST_JSON", "new_id", "utcnow",
           "sample_hit", "getenv"]
