"""Tracing — langsmith.run_helpers compatible (local-first + faster).

Why faster than langsmith?
- `PISMITH_TRACING=false` costs ~zero (a single bool check, no objects).
- One background batch writer (queue + single thread) instead of POST per run.
- Bounded `_safe` serialization (truncation + depth limit) + orjson when available.
- Sampling via `PISMITH_SAMPLE=0.1` to trace only 10% in production.
- Never blocks on network: optional best-effort endpoint with 2s timeout, in background.

Matching interface:
- @traceable / @trace
- trace(name=...) as a context manager
- tracing_context(enabled=..., tags=..., metadata=...)
- get_current_run_tree / current_run
- RunTree (name-compatible with langsmith RunTree)

Environment (new `PISMITH_*` names take precedence, legacy `PYSMITH_*` still honored):
- PISMITH_TRACING / PISMITH_STORE / PISMITH_BATCH / PISMITH_SAMPLE / PISMITH_ENDPOINT
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import queue
import threading
import time

from ._utils import fast_dumps, getenv, new_id, sample_hit

_ENABLED = getenv("PISMITH_TRACING", "PYSMITH_TRACING", default="true").lower() not in ("0", "false", "off", "no", "n")


def _store_path() -> str:
    return getenv("PISMITH_STORE", "PYSMITH_STORE", default=".pismith_runs.jsonl")


def _batch_n() -> int:
    try:
        return int(getenv("PISMITH_BATCH", "PYSMITH_BATCH", default="100") or 100)
    except Exception:
        return 100


def _sample_rate() -> float:
    try:
        return float(getenv("PISMITH_SAMPLE", "PYSMITH_SAMPLE", default="1.0") or 1.0)
    except Exception:
        return 1.0


_CURRENT_ATTR = "__pismith_traceable__"
_LEGACY_ATTR = "__pysmith_traceable__"  # accepted for backward compatibility


def _mark_traceable(fn):
    setattr(fn, _CURRENT_ATTR, True)
    setattr(fn, _LEGACY_ATTR, True)  # so old detectors keep working
    return fn


_current: contextvars.ContextVar = contextvars.ContextVar("pismith_run", default=None)
_tags_ctx: contextvars.ContextVar = contextvars.ContextVar("pismith_tags", default=None)
_meta_ctx: contextvars.ContextVar = contextvars.ContextVar("pismith_meta", default=None)
_enabled_ctx: contextvars.ContextVar = contextvars.ContextVar("pismith_enabled", default=None)

_Q: queue.Queue = queue.Queue(maxsize=20000)
_worker_started = False
_worker_lock = threading.Lock()


def is_enabled() -> bool:
    ov = _enabled_ctx.get()
    if ov is not None:
        return bool(ov)
    return _ENABLED


def set_enabled(v: bool):
    global _ENABLED
    _ENABLED = bool(v)


def current_run():
    return _current.get()


get_current_run_tree = current_run  # alias matching langsmith


def get_tracing_context() -> dict:
    return {"enabled": is_enabled(), "tags": _tags_ctx.get() or [],
            "metadata": _meta_ctx.get() or {}}


@contextlib.contextmanager
def tracing_context(*, enabled: bool | None = None, tags: list | None = None,
                    metadata: dict | None = None):
    """Match langsmith.tracing_context — disable/tag a whole section."""
    t1 = _enabled_ctx.set(enabled) if enabled is not None else None
    t2 = _tags_ctx.set(tags) if tags is not None else None
    t3 = _meta_ctx.set(metadata) if metadata is not None else None
    try:
        yield
    finally:
        if t1 is not None:
            _enabled_ctx.reset(t1)
        if t2 is not None:
            _tags_ctx.reset(t2)
        if t3 is not None:
            _meta_ctx.reset(t3)


def set_tracing_parent(parent_id: str | None):
    run = _current.get()
    if run is not None:
        run.parent_id = parent_id


def set_run_metadata(metadata: dict):
    run = _current.get()
    if run is not None:
        try:
            run.metadata.update(metadata)
        except Exception:
            pass


def is_traceable_function(fn) -> bool:
    return bool(getattr(fn, _CURRENT_ATTR, False) or getattr(fn, _LEGACY_ATTR, False))


def ensure_traceable(fn, **kw):
    if is_traceable_function(fn):
        return fn
    return traceable(fn, **kw)


def _safe(x, depth: int = 0):
    # Fast bounded serialization: prevents memory blowups on large inputs
    # (e.g. LangChain messages)
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return x if len(x) < 2000 else x[:2000] + "…"
    if depth > 3:
        return "…"
    if isinstance(x, dict):
        out = {}
        for i, (k, v) in enumerate(x.items()):
            if i >= 20:
                break
            try:
                out[str(k)[:80]] = _safe(v, depth + 1)
            except Exception:
                out["?"] = "?"
        return out
    if isinstance(x, (list, tuple)):
        return [_safe(v, depth + 1) for v in list(x)[:20]]
    if hasattr(x, "content"):  # BaseMessage from langchain
        try:
            return _safe(str(x.content)[:2000], depth + 1)
        except Exception:
            return "?"
    if hasattr(x, "to_dict"):
        try:
            return _safe(x.to_dict(), depth + 1)
        except Exception:
            pass
    try:
        s = str(x)
        return s[:2000]
    except Exception:
        return "?"


class Run:
    """Lightweight Run with __slots__ (faster to create than pydantic)."""
    __slots__ = ("id", "name", "run_type", "inputs", "outputs", "error",
                 "start", "end", "parent_id", "tags", "metadata", "children",
                 "feedback", "_events")

    def __init__(self, name: str, inputs=None, run_type: str = "chain",
                 tags: list | None = None, metadata: dict | None = None,
                 parent_id=None):
        self.id = new_id()
        self.name = name
        self.run_type = run_type
        self.inputs = inputs
        self.outputs = None
        self.error = None
        self.start = time.time()
        self.end = 0.0
        self.parent_id = parent_id
        ctx_tags = _tags_ctx.get() or []
        self.tags = list(tags or []) + list(ctx_tags)
        ctx_meta = _meta_ctx.get() or {}
        self.metadata = {**ctx_meta, **(metadata or {})}
        self.children: list[Run] = []
        self.feedback: list[dict] = []
        self._events: list[dict] = []

    @property
    def latency(self) -> float:
        return (self.end or time.time()) - self.start

    def add_feedback(self, key: str, score: float, comment: str = ""):
        self.feedback.append({"key": key, "score": score, "comment": comment})

    def dict(self) -> dict:
        return {"id": self.id, "name": self.name, "run_type": self.run_type,
                "inputs": _safe(self.inputs), "outputs": _safe(self.outputs),
                "error": str(self.error)[:500] if self.error else None,
                "latency": round(self.latency, 4), "parent_id": self.parent_id,
                "tags": self.tags, "metadata": self.metadata,
                "children": [c.dict() for c in self.children],
                "feedback": self.feedback}


class RunTree(Run):
    """Name-compatible with langsmith RunTree: create_child + local post/patch."""

    def create_child(self, name: str, inputs=None, run_type: str = "chain",
                     tags: list | None = None, metadata: dict | None = None) -> "RunTree":
        child = RunTree(name, inputs, run_type, tags, metadata, parent_id=self.id)
        self.children.append(child)
        return child

    def end(self, outputs=None, error=None):
        self.outputs = outputs if outputs is not None else self.outputs
        self.error = error if error is not None else self.error
        self.end = self.end or 0.0
        import time as _t
        if not self.end:
            self.end = _t.time()

    def post(self):
        _log_run(self)

    def patch(self):
        _log_run(self)


get_run_tree_context = current_run


class trace:
    """Match langsmith.run_helpers.trace — usable as decorator or context manager.

    @trace(name="step")  or  with trace(name="step", inputs={...}) as run:
    """

    def __init__(self, func=None, *, name: str | None = None,
                 run_type: str = "chain", tags: list | None = None,
                 metadata: dict | None = None, inputs=None):
        self._func = func
        self._name = name
        self._run_type = run_type
        self._tags = tags
        self._metadata = metadata
        self._inputs = inputs
        self._run: Run | None = None

    def __call__(self, *a, **k):
        if self._func is not None:  # used as bare @trace
            deco = traceable(self._func, name=self._name, run_type=self._run_type,
                             tags=self._tags, metadata=self._metadata)
            return deco(*a, **k)
        # used as @trace(name=...) -> return a decorator
        fn = a[0] if a else None
        if callable(fn) and not k:
            return traceable(fn, name=self._name, run_type=self._run_type,
                             tags=self._tags, metadata=self._metadata)
        raise TypeError("trace used incorrectly")

    def __enter__(self) -> Run:
        parent = _current.get()
        run = Run(self._name or "trace", self._inputs, self._run_type,
                  self._tags, self._metadata, parent.id if parent else None)
        self._tok = _current.set(run)
        self._run = run
        self._parent = parent
        return run

    def __exit__(self, exc_type, exc, tb):
        run = self._run
        assert run is not None
        run.end = time.time()
        if exc is not None:
            run.error = exc
        _current.reset(self._tok)
        parent = self._parent
        if parent is not None:
            parent.children.append(run)
        else:
            _log_run(run)
        return False


# --- Background: a single batch writer ---

def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_worker, daemon=True, name="pismith-writer")
        t.start()


def _worker():
    buf: list[dict] = []
    while True:
        try:
            item = _Q.get(timeout=2.0)
            if item is None:
                _flush(buf)
                return
            buf.append(item)
            while len(buf) < _batch_n():
                try:
                    nxt = _Q.get_nowait()
                    if nxt is None:
                        _flush(buf)
                        return
                    buf.append(nxt)
                except queue.Empty:
                    break
            _flush(buf)
            buf = []
        except queue.Empty:
            if buf:
                _flush(buf)
                buf = []


def _flush(buf: list[dict]):
    if not buf:
        return
    try:
        with open(_store_path(), "a", encoding="utf-8") as f:
            for r in buf:
                f.write(fast_dumps(r) + "\n")
    except Exception:
        pass
    ep = getenv("PISMITH_ENDPOINT", "PYSMITH_ENDPOINT")
    if ep:  # best-effort, never blocks
        try:
            import urllib.request
            body = fast_dumps(buf).encode()
            req = urllib.request.Request(ep, data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass


def _log_run(run: Run):
    if not is_enabled():
        return
    if not sample_hit(_sample_rate()):
        return
    _ensure_worker()
    try:
        _Q.put_nowait(run.dict())
    except Exception:
        pass


def _event(kind: str, name: str, data=None):
    if not is_enabled():
        return
    run = _current.get()
    if run is not None:
        try:
            run._events.append({"kind": kind, "name": name, "t": time.time()})
        except Exception:
            pass


def flush():
    """Wait for the queue to drain (call before exit/tests)."""
    if not is_enabled():
        return
    try:
        deadline = time.time() + 5.0
        while not _Q.empty() and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
    except Exception:
        pass


def traceable(func=None, *, name: str | None = None, run_type: str = "chain",
              tags: list | None = None, metadata: dict | None = None):
    """Decorator matching langsmith @traceable. Near-zero cost when disabled."""
    def deco(fn):
        fname = name or fn.__name__
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a, **k):
                if not is_enabled() or not sample_hit(_sample_rate()):
                    return await fn(*a, **k)
                parent = _current.get()
                run = Run(fname, {"args": _safe(a), "kwargs": _safe(k)},
                          run_type, tags, metadata, parent.id if parent else None)
                tok = _current.set(run)
                try:
                    out = await fn(*a, **k)
                    run.outputs = out
                    return out
                except Exception as e:
                    run.error = e
                    raise
                finally:
                    run.end = time.time()
                    _current.reset(tok)
                    if parent is not None:
                        parent.children.append(run)
                    else:
                        _log_run(run)
            _mark_traceable(aw)
            return aw

        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def gw(*a, **k):
                if not is_enabled():
                    yield from fn(*a, **k)
                    return
                parent = _current.get()
                run = Run(fname, {"args": _safe(a), "kwargs": _safe(k)},
                          run_type, tags, metadata, parent.id if parent else None)
                tok = _current.set(run)
                try:
                    out = yield from fn(*a, **k)
                    run.outputs = out
                    return out
                except Exception as e:
                    run.error = e
                    raise
                finally:
                    run.end = time.time()
                    _current.reset(tok)
                    if parent is not None:
                        parent.children.append(run)
                    else:
                        _log_run(run)
            _mark_traceable(gw)
            return gw

        @functools.wraps(fn)
        def w(*a, **k):
            if not is_enabled():
                return fn(*a, **k)
            if not sample_hit(_sample_rate()):
                return fn(*a, **k)
            parent = _current.get()
            run = Run(fname, {"args": _safe(a), "kwargs": _safe(k)},
                      run_type, tags, metadata, parent.id if parent else None)
            tok = _current.set(run)
            try:
                out = fn(*a, **k)
                run.outputs = out
                return out
            except Exception as e:
                run.error = e
                raise
            finally:
                run.end = time.time()
                _current.reset(tok)
                if parent is not None:
                    parent.children.append(run)
                else:
                    _log_run(run)
        _mark_traceable(w)
        return w
    return deco(func) if func else deco


__all__ = ["Run", "RunTree", "traceable", "trace", "current_run",
           "get_current_run_tree", "get_run_tree_context", "get_tracing_context",
           "tracing_context", "set_tracing_parent", "set_run_metadata",
           "is_traceable_function", "ensure_traceable", "flush",
           "is_enabled", "set_enabled"]
