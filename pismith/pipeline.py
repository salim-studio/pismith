"""Pipeline — simple traceable ETL/ML pipelines with optional caching.

    from pismith import Pipeline, step

    @step
    def load(ctx): return [{"x": 1}, {"x": 2}]

    @step
    def double(ctx, rows): return [{**r, "x2": r["x"]*2} for r in rows]

    pipe = Pipeline([load, double], name="etl-demo")
    out = pipe.run()
    print(pipe.summary())

- Every step is traced automatically (visible in runs).
- Optional per-step cache (speeds up re-runs).
- A metadata context flows through the steps.
"""
from __future__ import annotations

import time


class Step:
    def __init__(self, fn, name: str | None = None, cache: bool = False):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "step")
        self.cache = cache
        self._cached = None
        self._has_cache = False
        self.last_time_ms: float = 0.0

    def __call__(self, ctx: dict, data=None):
        if self.cache and self._has_cache:
            return self._cached
        t0 = time.time()
        run = None
        try:
            from . import tracing as _t
            if _t.is_enabled():
                run = _t.Run(f"pipe:{self.name}", {"step": self.name}, "chain")
        except Exception:
            run = None
        try:
            # A step takes (ctx) or (ctx, data)
            try:
                out = self.fn(ctx, data)
            except TypeError:
                out = self.fn(ctx) if data is None else self.fn(data)
            self.last_time_ms = round((time.time() - t0) * 1000, 2)
            if self.cache:
                self._cached = out
                self._has_cache = True
            if run is not None:
                try:
                    from . import tracing as _t
                    import time as _time
                    n = len(out) if isinstance(out, (list, dict)) else 1
                    run.outputs = {"n": n, "time_ms": self.last_time_ms}
                    run.end = _time.time()
                    _t._log_run(run)
                except Exception:
                    pass
            return out
        except Exception as e:
            if run is not None:
                try:
                    from . import tracing as _t
                    import time as _time
                    run.error = e
                    run.end = _time.time()
                    _t._log_run(run)
                except Exception:
                    pass
            raise

    def clear_cache(self):
        self._cached = None
        self._has_cache = False


def step(fn=None, *, name: str | None = None, cache: bool = False):
    """Use as @step or @step(cache=True)."""
    def deco(f):
        s = Step(f, name=name, cache=cache)
        # keep the name for compatibility
        s.__name__ = getattr(f, "__name__", "step")
        return s
    return deco(fn) if fn else deco


class Pipeline:
    def __init__(self, steps: list, name: str = "pipeline", context: dict | None = None):
        self.steps = [s if isinstance(s, Step) else Step(s) for s in steps]
        self.name = name
        self.context = dict(context or {})
        self.history: list[dict] = []

    def run(self, initial=None, context: dict | None = None):
        ctx = {**self.context, **(context or {})}
        data = initial
        self.history = []
        for s in self.steps:
            t0 = time.time()
            try:
                data = s(ctx, data)
                self.history.append({"step": s.name, "ok": True,
                                     "time_ms": round((time.time() - t0) * 1000, 2)})
            except Exception as e:
                self.history.append({"step": s.name, "ok": False, "error": str(e)[:300]})
                raise
        return data

    def summary(self) -> dict:
        total = sum(h.get("time_ms", 0) for h in self.history)
        return {"pipeline": self.name, "steps": len(self.steps),
                "total_ms": round(total, 2), "history": self.history}

    def clear_cache(self):
        for s in self.steps:
            s.clear_cache()

    def __repr__(self):
        return f"Pipeline({self.name!r}, steps={[s.name for s in self.steps]})"


__all__ = ["Step", "step", "Pipeline"]
