"""أدوات داخلية سريعة لـ pysmith: توليد ids + تسلسل JSON سريع + وقت."""
from __future__ import annotations

import json
import random
import time
import uuid

try:  # orjson أسرع ~5-10x إن وُجد، وإلا stdlib
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


__all__ = ["fast_dumps", "fast_loads", "FAST_JSON", "new_id", "utcnow", "sample_hit"]
