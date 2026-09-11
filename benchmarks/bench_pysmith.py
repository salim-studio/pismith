"""Benchmark: pysmith سرعة التتبع والتقييم (يعمل بـ python benchmarks/bench_pysmith.py)."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYSMITH_STORE"] = os.path.join(tempfile.gettempdir(), "bench_pysmith.jsonl")
try:
    os.remove(os.environ["PYSMITH_STORE"])
except OSError:
    pass

from pysmith import Client, evaluate, flush, traceable


def bench_disabled(n=20000):
    os.environ["PYSMITH_TRACING"] = "false"
    import importlib
    import pysmith.tracing as t
    t.set_enabled(False)

    @traceable(name="b")
    def f(x):
        return x + 1

    t0 = time.perf_counter()
    for i in range(n):
        f(i)
    dt = time.perf_counter() - t0
    print(f"disabled: {n} calls in {dt:.3f}s → {n/dt:,.0f} call/s ({dt/n*1e6:.2f} µs/call)")
    t.set_enabled(True)


def bench_enabled(n=5000):
    from pysmith import tracing as t
    t.set_enabled(True)

    @traceable(name="bench")
    def f(x):
        return x * 2

    t0 = time.perf_counter()
    for i in range(n):
        f(i)
    flush()
    dt = time.perf_counter() - t0
    size = os.path.getsize(os.environ["PYSMITH_STORE"])
    print(f"enabled+batch-write: {n} runs in {dt:.3f}s → {n/dt:,.0f} run/s (file {size/1024:.1f} KB)")


def bench_evaluate(n=200):
    with tempfile.TemporaryDirectory() as d:
        c = Client(base_dir=os.path.join(d, "db"))
        c.create_dataset("b")
        c.create_examples("b", [{"inputs": {"q": str(i)}, "outputs": {"a": str(i)}} for i in range(n)])
        for w in (1, 8):
            t0 = time.perf_counter()
            r = evaluate(lambda inp: {"a": inp["q"]}, "b", client=c, max_workers=w)
            dt = time.perf_counter() - t0
            print(f"evaluate n={n} workers={w}: {dt:.3f}s → {n/dt:,.0f} ex/s (errors={r.summary['errors']})")


if __name__ == "__main__":
    bench_disabled()
    bench_enabled()
    bench_evaluate()
