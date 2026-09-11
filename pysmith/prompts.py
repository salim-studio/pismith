"""Prompts hub محلي — مطابق لسطح langsmith prompts (create/get/list)."""
from __future__ import annotations

import os
import time

from ._utils import fast_dumps, fast_loads, new_id


def _path(base_dir: str) -> str:
    return os.path.join(base_dir, "prompts.json")


def _load(base_dir: str) -> dict:
    p = _path(base_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return fast_loads(f.read() or "{}")
    except Exception:
        return {}


def _save(base_dir: str, d: dict):
    os.makedirs(base_dir, exist_ok=True)
    tmp = _path(base_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(fast_dumps(d))
    os.replace(tmp, _path(base_dir))


def create_prompt(name: str, template: str, base_dir: str = ".pysmith",
                  variables: list | None = None) -> dict:
    d = _load(base_dir)
    hist = d.get(name, [])
    commit = f"v{len(hist) + 1}-{new_id(6)}"
    entry = {"name": name, "template": template, "commit": commit,
             "variables": variables or [], "created_at": time.time()}
    hist.append(entry)
    d[name] = hist
    _save(base_dir, d)
    return dict(entry)


def get_prompt(name: str, base_dir: str = ".pysmith", commit: str | None = None) -> dict:
    hist = _load(base_dir).get(name, [])
    if not hist:
        raise KeyError(f"prompt not found: {name}")
    if commit:
        for e in hist:
            if e.get("commit") == commit:
                return dict(e)
        raise KeyError(f"commit not found: {commit}")
    return dict(hist[-1])


def list_prompts(base_dir: str = ".pysmith", limit: int = 50) -> list[dict]:
    d = _load(base_dir)
    out = [v[-1] for v in d.values() if v]
    return [dict(e) for e in out[:limit]]


def format_prompt(name: str, base_dir: str = ".pysmith", **vars) -> str:
    p = get_prompt(name, base_dir)
    t = p["template"]
    try:
        return t.format(**vars)
    except Exception:
        return t


__all__ = ["create_prompt", "get_prompt", "list_prompts", "format_prompt"]
