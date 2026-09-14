"""Experiments — lightweight ML experiment tracking (local-first MLflow alternative).

    from pismith import Experiments
    xp = Experiments()  # .pismith_experiments/
    xp.create_experiment("churn")
    with xp.run("churn", params={"lr": 0.01}, tags=["v1"]) as r:
        xp.log_metric("accuracy", 0.93)
        # ... train your model here ...
    print(xp.best_run("churn", metric="accuracy"))
    print(xp.compare("churn"))          # analysis-ready list[dict]
    xp.to_csv("churn", "runs.csv")

- Atomic JSONL storage + cache.
- Zero dependencies, integrated with tracing (each experiment run is a traced run).
- Conceptually MLflow-compatible: experiment/run/params/metrics/artifacts.
"""
from __future__ import annotations

import os
import time
from threading import RLock

from ._utils import fast_dumps, fast_loads, new_id


class _RunCtx:
    def __init__(self, tracker: "Experiments", exp: str, name: str, params: dict, tags: list):
        self.t = tracker
        self.exp = exp
        self.id = new_id()
        self.name = name
        self.params = dict(params or {})
        self.tags = list(tags or [])
        self.metrics: dict = {}
        self.artifacts: list = []
        self.t0 = time.time()
        self._trace = None

    def __enter__(self) -> "_RunCtx":
        try:
            from . import tracing as _t
            if _t.is_enabled():
                self._trace = _t.Run(f"exp:{self.exp}/{self.name}",
                                     {"params": self.params}, "experiment",
                                     tags=self.tags)
        except Exception:
            self._trace = None
        return self

    def log_metric(self, k: str, v: float, step: int | None = None):
        self.metrics[k] = {"value": float(v), "step": step, "t": time.time()}

    def log_metrics(self, d: dict):
        for k, v in d.items():
            self.log_metric(k, v)

    def log_param(self, k: str, v):
        self.params[k] = v

    def log_artifact(self, path: str):
        self.artifacts.append(path)

    def __exit__(self, exc_type, exc, tb):
        latency = time.time() - self.t0
        rec = {"id": self.id, "experiment": self.exp, "name": self.name,
               "params": self.params,
               "metrics": {k: v["value"] for k, v in self.metrics.items()},
               "metrics_full": self.metrics, "tags": self.tags,
               "artifacts": self.artifacts, "latency": round(latency, 4),
               "error": str(exc)[:500] if exc else None, "t": time.time()}
        self.t._append(self.exp, rec)
        if self._trace is not None:
            try:
                from . import tracing as _t
                import time as _time
                self._trace.outputs = rec["metrics"]
                self._trace.error = exc
                self._trace.end = _time.time()
                _t._log_run(self._trace)
            except Exception:
                pass
        return False


class Experiments:
    def __init__(self, base_dir: str = ".pismith_experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._lock = RLock()
        self._cache: dict[str, list[dict]] = {}

    def _path(self, exp: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in exp)
        return os.path.join(self.base_dir, f"{safe}.jsonl")

    def create_experiment(self, name: str, description: str = "") -> dict:
        meta = os.path.join(self.base_dir, "experiments.json")
        try:
            with open(meta, encoding="utf-8") as f:
                d = fast_loads(f.read() or "{}")
        except Exception:
            d = {}
        d[name] = {"name": name, "description": description, "created_at": time.time()}
        tmp = meta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(fast_dumps(d))
        os.replace(tmp, meta)
        if not os.path.exists(self._path(name)):
            open(self._path(name), "a").close()
        return d[name]

    def list_experiments(self) -> list[str]:
        return [f[:-6] for f in os.listdir(self.base_dir) if f.endswith(".jsonl")]

    def run(self, experiment: str, name: str | None = None, params: dict | None = None,
            tags: list | None = None) -> _RunCtx:
        self.create_experiment(experiment)
        return _RunCtx(self, experiment, name or f"run-{new_id(6)}", params or {}, tags or [])

    # Fast functional alternative: xp.log_run("exp", params, metrics)
    def log_run(self, experiment: str, params: dict, metrics: dict,
                name: str | None = None, tags: list | None = None) -> dict:
        with self.run(experiment, name, params, tags) as r:
            r.log_metrics(metrics)
            rid = r.id
        return self.get_run(experiment, rid)

    def _append(self, exp: str, rec: dict):
        with self._lock:
            with open(self._path(exp), "a", encoding="utf-8") as f:
                f.write(fast_dumps(rec) + "\n")
            self._cache.pop(exp, None)

    def list_runs(self, experiment: str, limit: int = 1000) -> list[dict]:
        if experiment in self._cache:
            return self._cache[experiment][-limit:]
        p = self._path(experiment)
        if not os.path.exists(p):
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(fast_loads(line))
                except Exception:
                    continue
        self._cache[experiment] = out
        return out[-limit:]

    def get_run(self, experiment: str, run_id: str) -> dict:
        for r in self.list_runs(experiment, limit=100000):
            if r.get("id") == run_id:
                return r
        raise KeyError(run_id)

    def best_run(self, experiment: str, metric: str, mode: str = "max") -> dict | None:
        runs = [r for r in self.list_runs(experiment) if metric in (r.get("metrics") or {})]
        if not runs:
            return None
        key = lambda r: r["metrics"][metric]
        return max(runs, key=key) if mode == "max" else min(runs, key=key)

    def compare(self, experiment: str, limit: int = 200) -> list[dict]:
        """Flat rows {run, param:*, metric:*} — ready for DataTable/pandas."""
        rows = []
        for r in self.list_runs(experiment, limit):
            d = {"run": r.get("name"), "id": r.get("id"), "latency": r.get("latency")}
            for k, v in (r.get("params") or {}).items():
                d[f"param:{k}"] = v
            for k, v in (r.get("metrics") or {}).items():
                d[f"metric:{k}"] = v
            rows.append(d)
        return rows

    def to_csv(self, experiment: str, path: str) -> str:
        from .data import DataTable
        return DataTable(self.compare(experiment)).to_csv(path)

    def delete_experiment(self, experiment: str):
        p = self._path(experiment)
        if os.path.exists(p):
            os.remove(p)
        self._cache.pop(experiment, None)


__all__ = ["Experiments"]
