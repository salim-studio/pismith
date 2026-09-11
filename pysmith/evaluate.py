"""Evaluate — مطابق لـ langsmith.evaluate (متوازٍ + أسرع).

السرعة:
- ThreadPoolExecutor متوازٍ (I/O-bound: استدعاءات LLM)
- دعم async عبر aevaluate (asyncio.gather + Semaphore)
- ملخص إحصائي واحد (avg/p50/p95 + متوسط كل score)
"""
from __future__ import annotations

import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor


class ExperimentResults:
    __slots__ = ("experiment", "results", "summary")

    def __init__(self, experiment: str, results: list[dict]):
        self.experiment = experiment
        self.results = results
        lat = [r.get("latency", 0) or 0 for r in results]
        scores: dict[str, list[float]] = {}
        for r in results:
            for k, v in (r.get("scores") or {}).items():
                if isinstance(v, (int, float)):
                    scores.setdefault(k, []).append(float(v))
        s = sorted(lat)
        def pct(p):
            if not s:
                return 0
            i = min(len(s) - 1, int(p * len(s)))
            return round(s[i], 4)
        self.summary = {
            "n": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "avg_latency": round(sum(lat) / len(lat), 4) if lat else 0,
            "p50_latency": pct(0.5),
            "p95_latency": pct(0.95),
            "scores": {k: round(sum(v) / len(v), 4) for k, v in scores.items()},
        }

    def __repr__(self):
        return f"ExperimentResults({self.experiment!r}, {self.summary})"

    def to_dict(self) -> dict:
        return {"experiment": self.experiment, "results": self.results,
                "summary": self.summary}


def _score_one(evaluators, example: dict, outputs) -> dict:
    scores: dict = {}
    for ev in evaluators or []:
        try:
            r = ev(example, outputs) if callable(ev) else None
            if isinstance(r, dict):
                scores.update(r)
            elif isinstance(r, bool):
                scores[getattr(ev, "__name__", "score")] = 1.0 if r else 0.0
            elif isinstance(r, (int, float)):
                scores[getattr(ev, "__name__", "score")] = float(r)
        except Exception as e:
            scores[getattr(ev, "__name__", "eval") + "_error"] = str(e)[:200]
    return scores


def _run_one(target, evaluators, example: dict) -> dict:
    t0 = time.time()
    try:
        out = target(example.get("inputs", example))
        if asyncio.iscoroutine(out):
            out = asyncio.run(out)
        return {"example_id": example.get("id"), "outputs": out,
                "scores": _score_one(evaluators, example, out),
                "latency": round(time.time() - t0, 4), "error": None}
    except Exception as e:
        return {"example_id": example.get("id"), "outputs": None, "scores": {},
                "latency": round(time.time() - t0, 4), "error": str(e)[:500]}


def _resolve_examples(dataset, client) -> list[dict]:
    if isinstance(dataset, str):
        from .client import Client as _C
        c = client or _C()
        return c.list_examples(dataset)
    if isinstance(dataset, list):
        return dataset
    if hasattr(dataset, "examples"):
        return list(dataset.examples)
    if dataset is None:
        return []
    return list(dataset)


def evaluate(target, dataset=None, evaluators: list | None = None,
             experiment: str = "exp-1", max_workers: int = 8,
             client=None, **kw) -> ExperimentResults:
    """target: دالة(inputs)->outputs. dataset: اسم أو قائمة أمثلة."""
    examples = _resolve_examples(dataset, client)
    if not examples:
        return ExperimentResults(experiment, [])
    n = max(1, min(max_workers, len(examples)))
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda e: _run_one(target, evaluators, e), examples))
    res = ExperimentResults(experiment, results)
    # سجّل التجربة كـ feedback/run محلي اختياري
    if client is not None and kw.get("log", False):
        for r in results:
            try:
                client.create_feedback(str(r.get("example_id") or experiment),
                                       experiment, r["scores"].get("score", 0) if r["scores"] else 0)
            except Exception:
                pass
    return res


async def aevaluate(target, dataset=None, evaluators: list | None = None,
                    experiment: str = "exp-1", max_concurrency: int = 8,
                    client=None, **kw) -> ExperimentResults:
    """نسخة async مطابقة لـ langsmith.aevaluate."""
    examples = _resolve_examples(dataset, client)
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def one(ex: dict) -> dict:
        async with sem:
            t0 = time.time()
            try:
                out = target(ex.get("inputs", ex))
                if asyncio.iscoroutine(out):
                    out = await out
                elif callable(getattr(out, "__await__", None)):
                    out = await out
                scores = _score_one(evaluators, ex, out)
                return {"example_id": ex.get("id"), "outputs": out, "scores": scores,
                        "latency": round(time.time() - t0, 4), "error": None}
            except Exception as e:
                return {"example_id": ex.get("id"), "outputs": None, "scores": {},
                        "latency": round(time.time() - t0, 4), "error": str(e)[:500]}

    results = await asyncio.gather(*(one(e) for e in examples))
    return ExperimentResults(experiment, list(results))


def evaluate_run(run: dict, evaluators: list | None = None) -> dict:
    """قيّم run واحداً جاهزاً (مطابق لـ langsmith.evaluate_run)."""
    example = {"inputs": run.get("inputs"), "outputs": run.get("outputs"),
               "id": run.get("id")}
    return {"run_id": run.get("id"),
            "scores": _score_one(evaluators, example, run.get("outputs"))}


# ---------- مقيّمات جاهزة (مطابقة evaluators في langsmith) ----------

def exact_match(example: dict, outputs) -> dict:
    exp = example.get("outputs")
    if isinstance(exp, dict):
        exp = next(iter(exp.values()), "")
    got = outputs.get("output", outputs) if isinstance(outputs, dict) else outputs
    if isinstance(got, dict):
        got = next(iter(got.values()), "")
    return {"exact_match": 1.0 if str(got).strip() == str(exp).strip() else 0.0}


def contains(expected_key: str = "answer"):
    def _ev(example: dict, outputs) -> dict:
        o = example.get("outputs")
        exp = o.get(expected_key, "") if isinstance(o, dict) else o
        got = str(outputs)
        e = str(exp or "")
        return {"contains": 1.0 if (e and (e[:60] in got or e.strip() in got)) else 0.0}
    _ev.__name__ = "contains"
    return _ev


def regex_match(pattern: str, key: str = "output"):
    import re
    rx = re.compile(pattern)

    def _ev(example: dict, outputs) -> dict:
        got = outputs.get(key, outputs) if isinstance(outputs, dict) else outputs
        return {"regex_match": 1.0 if rx.search(str(got or "")) else 0.0}
    _ev.__name__ = "regex_match"
    return _ev


def score_string(expected: str, key: str = "output"):
    """مطابق تقريبي: 1.0 تطابق، 0.5 احتواء، 0.0 غير ذلك."""
    def _ev(example: dict, outputs) -> dict:
        got = outputs.get(key, outputs) if isinstance(outputs, dict) else outputs
        g, e = str(got or "").strip(), str(expected).strip()
        if g == e:
            return {"score_string": 1.0}
        if e and e in g:
            return {"score_string": 0.5}
        return {"score_string": 0.0}
    _ev.__name__ = "score_string"
    return _ev


__all__ = ["evaluate", "aevaluate", "evaluate_run", "ExperimentResults",
           "exact_match", "contains", "regex_match", "score_string"]
