"""Client — langsmith.Client-compatible surface (local-first: JSON + fast lookup).

Local storage:
  .pismith/datasets.json   → datasets + examples
  .pismith_runs.jsonl      → runs (via tracing)
  .pismith/feedback.jsonl  → feedback
  .pismith/prompts.json    → prompts hub

Speed:
- in-memory cache + atomic writes (tmp + os.replace)
- a single RLock instead of network locking
- list_runs reads only the last N lines (never loads the whole file)
- batch_ingest_runs writes in one batch
"""
from __future__ import annotations

import os
import time
from collections import deque
from threading import RLock

from ._utils import fast_dumps, fast_loads, getenv, new_id
from .schemas import Dataset


class Client:
    def __init__(self, base_dir: str = ".pismith", api_key: str | None = None,
                 endpoint: str | None = None, api_url: str | None = None):
        self.base_dir = base_dir
        self.api_key = api_key or getenv("PISMITH_API_KEY", "PYSMITH_API_KEY",
                                         "LANGSMITH_API_KEY")
        self.endpoint = (endpoint or api_url
                         or getenv("PISMITH_ENDPOINT", "PYSMITH_ENDPOINT")).rstrip("/")
        self.api_url = self.endpoint
        self._lock = RLock()
        os.makedirs(base_dir, exist_ok=True)
        self._ds_path = os.path.join(base_dir, "datasets.json")
        self._fb_path = os.path.join(base_dir, "feedback.jsonl")
        self._cache: dict | None = None
        if not os.path.exists(self._ds_path):
            with open(self._ds_path, "w", encoding="utf-8") as f:
                f.write("{}")

    # ---------- helpers ----------
    def _load_ds(self) -> dict:
        if self._cache is not None:
            return self._cache
        try:
            with open(self._ds_path, encoding="utf-8") as f:
                self._cache = fast_loads(f.read() or "{}")
        except Exception:
            self._cache = {}
        return self._cache

    def _save_ds(self, d: dict):
        self._cache = d
        tmp = self._ds_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(fast_dumps(d))
        os.replace(tmp, self._ds_path)

    def _runs_path(self) -> str:
        return getenv("PISMITH_STORE", "PYSMITH_STORE", default=".pismith_runs.jsonl")

    # ---------- datasets (langsmith-compatible) ----------
    def create_dataset(self, dataset_name: str, description: str = "",
                       data_type: str = "kv", **kw) -> dict:
        name = dataset_name
        with self._lock:
            d = dict(self._load_ds())
            if name not in d:
                d[name] = {"id": new_id(), "name": name, "description": description,
                           "data_type": data_type, "examples": [], "created_at": time.time()}
                self._save_ds(d)
            return dict(d[name])

    def read_dataset(self, dataset_name: str | None = None, dataset_id: str | None = None) -> dict:
        d = self._load_ds()
        if dataset_name and dataset_name in d:
            return dict(d[dataset_name])
        if dataset_id:
            for v in d.values():
                if v.get("id") == dataset_id:
                    return dict(v)
        raise KeyError(f"dataset not found: {dataset_name or dataset_id}")

    def has_dataset(self, dataset_name: str) -> bool:
        return dataset_name in self._load_ds()

    def list_datasets(self, limit: int = 100) -> list[dict]:
        vals = list(self._load_ds().values())
        return [dict(v) for v in vals[:limit]]

    datasets = list_datasets

    def delete_dataset(self, dataset_name: str | None = None, dataset_id: str | None = None):
        with self._lock:
            d = dict(self._load_ds())
            if dataset_name and dataset_name in d:
                del d[dataset_name]
            elif dataset_id:
                for k, v in list(d.items()):
                    if v.get("id") == dataset_id:
                        del d[k]
                        break
            self._save_ds(d)

    # langsmith projects alias (a local project behaves like a dataset group)
    create_project = create_dataset
    read_project = read_dataset
    has_project = has_dataset
    list_projects = list_datasets
    delete_project = delete_dataset

    # ---------- examples ----------
    def create_example(self, inputs: dict, outputs: dict | None = None,
                       dataset_name: str | None = None, dataset_id: str | None = None,
                       metadata: dict | None = None, **kw) -> dict:
        return self.create_examples(dataset_name or dataset_id or "default",
                                    [{"inputs": inputs, "outputs": outputs,
                                      "metadata": metadata or {}}])[0]

    def create_examples(self, dataset_name: str, examples: list[dict]) -> list[dict]:
        """examples: [{inputs, outputs?, metadata?, id?}] — also accepts a dataset_id."""
        with self._lock:
            d = dict(self._load_ds())
            ds = d.get(dataset_name)
            if ds is None:  # جرّب id
                for v in d.values():
                    if v.get("id") == dataset_name:
                        ds = v
                        dataset_name = v["name"]
                        break
            if ds is None:
                ds = {"id": new_id(), "name": dataset_name, "description": "",
                      "examples": [], "created_at": time.time()}
                d[dataset_name] = ds
            out = []
            for e in examples:
                ex = {"id": e.get("id") or new_id(),
                      "inputs": e.get("inputs", {}),
                      "outputs": e.get("outputs"),
                      "metadata": e.get("metadata") or {},
                      "created_at": time.time()}
                ds["examples"].append(ex)
                out.append(dict(ex))
            self._save_ds(d)
            return out

    # compat: create_example_from_run / create_llm_example / create_chat_example
    def create_example_from_run(self, run: dict, dataset_name: str, **kw) -> dict:
        return self.create_example(run.get("inputs") or {}, run.get("outputs"),
                                   dataset_name=dataset_name)[0] if False else \
            self.create_examples(dataset_name, [{"inputs": run.get("inputs") or {},
                                                 "outputs": run.get("outputs")}])[0]

    def create_llm_example(self, prompt: str, response: str, dataset_name: str, **kw) -> dict:
        return self.create_examples(dataset_name, [{"inputs": {"prompt": prompt},
                                                    "outputs": {"response": response}}])[0]

    create_chat_example = create_llm_example

    def read_example(self, example_id: str) -> dict:
        for ds in self._load_ds().values():
            for e in ds.get("examples", []):
                if e.get("id") == example_id:
                    return dict(e)
        raise KeyError(example_id)

    def list_examples(self, dataset_name: str | None = None, dataset_id: str | None = None,
                      limit: int = 10000, **kw) -> list[dict]:
        d = self._load_ds()
        ds = None
        if dataset_name and dataset_name in d:
            ds = d[dataset_name]
        elif dataset_id:
            for v in d.values():
                if v.get("id") == dataset_id:
                    ds = v
                    break
        elif dataset_name is None and dataset_id is None:
            out = []
            for v in d.values():
                out.extend(v.get("examples", []))
            return [dict(e) for e in out[:limit]]
        if not ds:
            return []
        return [dict(e) for e in ds.get("examples", [])[:limit]]

    def update_example(self, example_id: str, inputs: dict | None = None,
                       outputs: dict | None = None, metadata: dict | None = None) -> dict:
        with self._lock:
            d = dict(self._load_ds())
            for ds in d.values():
                for e in ds.get("examples", []):
                    if e.get("id") == example_id:
                        if inputs is not None:
                            e["inputs"] = inputs
                        if outputs is not None:
                            e["outputs"] = outputs
                        if metadata is not None:
                            e["metadata"] = metadata
                        self._save_ds(d)
                        return dict(e)
        raise KeyError(example_id)

    def delete_example(self, example_id: str):
        with self._lock:
            d = dict(self._load_ds())
            for ds in d.values():
                ex = ds.get("examples", [])
                for i, e in enumerate(ex):
                    if e.get("id") == example_id:
                        ex.pop(i)
                        self._save_ds(d)
                        return

    def delete_examples(self, example_ids: list[str]):
        s = set(example_ids)
        with self._lock:
            d = dict(self._load_ds())
            for ds in d.values():
                ds["examples"] = [e for e in ds.get("examples", []) if e.get("id") not in s]
            self._save_ds(d)

    # ---------- runs ----------
    def _iter_run_lines(self, limit: int):
        path = self._runs_path()
        if not os.path.exists(path):
            return []
        # efficiently read the last N lines: tail via deque
        with open(path, encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
        out = []
        for ln in lines:
            try:
                out.append(fast_loads(ln))
            except Exception:
                continue
        return out

    def list_runs(self, limit: int = 50, project_name: str | None = None,
                 run_type: str | None = None, **kw) -> list[dict]:
        runs = self._iter_run_lines(max(1, limit * 2 if project_name or run_type else limit))
        if project_name:
            runs = [r for r in runs if r.get("name") == project_name or
                    project_name in (r.get("tags") or [])]
        if run_type:
            runs = [r for r in runs if r.get("run_type") == run_type]
        return runs[-limit:]

    runs = None  # kept as list_runs below; langsmith exposes .runs as a namespace

    def read_run(self, run_id: str) -> dict:
        for r in self._iter_run_lines(5000):
            if r.get("id") == run_id:
                return r
        raise KeyError(run_id)

    def create_run(self, name: str, inputs: dict | None = None, run_type: str = "chain",
                   **kw) -> dict:
        from . import tracing as _t
        run = _t.Run(name, inputs, run_type, kw.get("tags"), kw.get("metadata"))
        if kw.get("outputs") is not None:
            run.outputs = kw["outputs"]
        run.end = time.time()
        _t._log_run(run)
        return run.dict()

    def batch_ingest_runs(self, runs: list[dict]):
        from . import tracing as _t
        for r in runs:
            try:
                _t._Q.put_nowait(r if isinstance(r, dict) else r.dict())
            except Exception:
                break
        _t._ensure_worker()

    multipart_ingest = batch_ingest_runs

    def get_run_url(self, run: dict | str) -> str:
        rid = run.get("id") if isinstance(run, dict) else run
        return f"pismith://run/{rid}"

    def get_run_stats(self, **kw) -> dict:
        runs = self.list_runs(limit=kw.get("limit", 200))
        lat = [r.get("latency", 0) or 0 for r in runs]
        return {"count": len(runs),
                "avg_latency": round(sum(lat) / len(lat), 4) if lat else 0,
                "errors": sum(1 for r in runs if r.get("error"))}

    # ---------- feedback ----------
    def create_feedback(self, run_id: str, key: str, score: float | None = None,
                        comment: str | None = None, **kw) -> dict:
        fb = {"id": new_id(), "run_id": run_id, "key": key, "score": score,
              "comment": comment or "", "t": time.time()}
        with self._lock:
            with open(self._fb_path, "a", encoding="utf-8") as f:
                f.write(fast_dumps(fb) + "\n")
        return fb

    def list_feedback(self, run_ids: list[str] | None = None, limit: int = 100) -> list[dict]:
        if not os.path.exists(self._fb_path):
            return []
        out = []
        with open(self._fb_path, encoding="utf-8") as f:
            for line in f:
                try:
                    fb = fast_loads(line)
                except Exception:
                    continue
                if run_ids is None or fb.get("run_id") in run_ids:
                    out.append(fb)
        return out[-limit:]

    def read_feedback(self, feedback_id: str) -> dict:
        for fb in self.list_feedback(limit=100000):
            if fb.get("id") == feedback_id:
                return fb
        raise KeyError(feedback_id)

    def delete_feedback(self, feedback_id: str):
        if not os.path.exists(self._fb_path):
            return
        with self._lock:
            kept = [fb for fb in self.list_feedback(limit=1000000)
                    if fb.get("id") != feedback_id]
            with open(self._fb_path, "w", encoding="utf-8") as f:
                for fb in kept:
                    f.write(fast_dumps(fb) + "\n")

    # ---------- evaluate shortcuts (matching langsmith.Client.evaluate) ----------
    def evaluate(self, target, dataset: str | list, evaluators: list | None = None,
                 experiment: str = "exp-1", **kw):
        from .evaluate import evaluate as _ev
        return _ev(target, dataset, evaluators, experiment, client=self, **kw)

    def aevaluate(self, *a, **k):
        from .evaluate import aevaluate as _aev
        return _aev(*a, **k, client=self)

    # ---------- local prompts hub ----------
    def create_prompt(self, name: str, template: str, **kw) -> dict:
        from .prompts import create_prompt as _cp
        return _cp(name, template, base_dir=self.base_dir)

    def get_prompt(self, name: str, **kw) -> dict:
        from .prompts import get_prompt as _gp
        return _gp(name, base_dir=self.base_dir)

    def list_prompts(self, limit: int = 50) -> list[dict]:
        from .prompts import list_prompts as _lp
        return _lp(base_dir=self.base_dir, limit=limit)

    pull_prompt = get_prompt

    def push_prompt(self, name: str, template: str, **kw) -> dict:
        return self.create_prompt(name, template)

    def flush(self):
        from .tracing import flush as _flush
        _flush()

    def close(self):
        self.flush()

    # ---------- misc compat ----------
    def info(self) -> dict:
        return {"backend": "pismith-local", "base_dir": self.base_dir,
                "datasets": len(self._load_ds())}

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


# note: langsmith exposes .runs as a namespace; we keep list_runs only
# (runs=None is deleted so dir(Client) stays clean)
delattr(Client, "runs")

__all__ = ["Client"]
