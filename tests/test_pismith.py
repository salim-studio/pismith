"""Tests for pismith — langsmith compatibility + speed."""
import asyncio
import os
import tempfile

import pismith as ps
from pismith import (Client, contains, evaluate, exact_match, trace,
                     traceable, tracing_context, get_current_run_tree, flush)


def test_traceable_sync_async_and_nesting():
    @traceable(name="demo")
    def f(x: int) -> int:
        return x * 2

    @traceable(name="outer")
    def g(x: int) -> int:
        return f(x) + f(x + 1)

    assert f(21) == 42
    assert g(1) == 2 + 4

    @traceable(name="ad")
    async def af(x: int) -> int:
        return x + 1

    assert asyncio.run(af(1)) == 2


def test_trace_context_manager_and_tracing_context():
    with trace(name="step", inputs={"a": 1}) as run:
        assert run.name == "step"
        assert get_current_run_tree() is run
    assert get_current_run_tree() is None

    @traceable(name="inner")
    def h():
        return 1

    with tracing_context(enabled=False):
        assert h() == 1  # no recording, no error
    assert h() == 1


def test_disabled_zero_overhead():
    ps.tracing.set_enabled(False)
    try:
        @traceable
        def f(x):
            return x
        assert f(5) == 5
    finally:
        ps.tracing.set_enabled(True)


def test_client_datasets_examples_feedback_runs():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "runs.jsonl")
        os.environ["PISMITH_STORE"] = store
        try:
            c = Client(base_dir=os.path.join(d, "db"))
            c.create_dataset("ds", description="test")
            assert c.has_dataset("ds")
            assert c.read_dataset("ds")["name"] == "ds"
            ex = c.create_examples("ds", [{"inputs": {"q": "a"}, "outputs": {"a": "a"}},
                                          {"inputs": {"q": "b"}, "outputs": {"a": "b"}}])
            assert len(ex) == 2
            assert len(c.list_examples("ds")) == 2
            eid = ex[0]["id"]
            assert c.read_example(eid)["id"] == eid
            c.update_example(eid, metadata={"k": 1})
            assert c.read_example(eid)["metadata"] == {"k": 1}

            # runs via traceable
            @traceable(name="r1")
            def f(x):
                return x
            f(1)
            flush()
            runs = c.list_runs(limit=10)
            assert runs and runs[-1]["name"] == "r1"
            assert c.read_run(runs[-1]["id"])["id"] == runs[-1]["id"]

            fb = c.create_feedback(runs[-1]["id"], "correct", 1.0)
            assert fb["run_id"] == runs[-1]["id"]
            assert c.list_feedback([runs[-1]["id"]])

            # prompts hub
            c.create_prompt("greet", "hello {name}")
            assert "hello" in c.get_prompt("greet")["template"]
            assert c.list_prompts()

            # evaluate (sync + async)
            res = evaluate(lambda inp: {"a": inp["q"]}, "ds",
                           evaluators=[exact_match], client=c)
            assert res.summary["n"] == 2
            assert res.summary["scores"]["exact_match"] == 1.0

            ares = asyncio.run(c.aevaluate(lambda inp: {"a": inp["q"]}, "ds",
                                           evaluators=[exact_match]))
            assert ares.summary["n"] == 2

            # extra evaluators
            r2 = evaluate(lambda inp: "hello world", [{"inputs": {}, "outputs": {"output": "world"}}],
                          evaluators=[contains("output")])
            assert r2.summary["n"] == 1

            # delete
            c.delete_example(eid)
            assert len(c.list_examples("ds")) == 1
            c.delete_dataset("ds")
            assert not c.has_dataset("ds")
        finally:
            os.environ.pop("PISMITH_STORE", None)


def test_langsmith_api_parity_names():
    # key langsmith names exist on pismith.Client
    for m in ["create_dataset", "read_dataset", "has_dataset", "list_datasets",
              "delete_dataset", "create_example", "create_examples",
              "read_example", "list_examples", "update_example", "delete_example",
              "list_runs", "read_run", "create_run", "batch_ingest_runs",
              "create_feedback", "list_feedback", "evaluate",
              "create_prompt", "get_prompt", "list_prompts", "flush"]:
        assert hasattr(Client, m), m
    for m in ["traceable", "trace", "tracing_context", "get_current_run_tree"]:
        assert hasattr(ps, m), m
