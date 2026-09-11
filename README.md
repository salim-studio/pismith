# pysmith — بديل LangSmith الأسرع (محلي أولاً)

```bash
pip install git+https://github.com/salim-studio/pysmith.git
# أو: pip install -e .  (من جذر المستودع)
# للسرعة القصوى: pip install orjson
```

مكتبة مطابقة لواجهة `langsmith` ولكن أسرع: تتبع محلي batch + تقييم متوازٍ + صفر تكلفة عند التعطيل.

## لماذا أسرع؟

| langsmith | pysmith |
|---|---|
| POST شبكي متزامن لكل run (يحجب + يحتاج API key) | كاتب خلفية واحد batch (queue + thread) + ملف JSONL محلي |
| pydantic ثقيل لكل Run | `Run` بـ `__slots__` + تسلسل محدود `_safe` (قص 2000 حرف/عمق 3) |
| لا يوجد مسار صفر-تكلفة | `PYSMITH_TRACING=false` → فحص `bool` واحد (~1.3µs/call) |
| evaluate متسلسل غالباً | `evaluate` متوازٍ (ThreadPool) + `aevaluate` (asyncio) |
| orjson/zstd اختياري معقد | orjson تلقائياً إن وُجد، وإلا stdlib — بلا اعتماديات إجبارية |

أرقام حقيقية (Windows، Python 3.14 — `benchmarks/bench_pysmith.py`):
- معطّل: **~760k استدعاء/ثانية (~1.3µs)** — تكلفة شبه صفرية.
- مفعّل + كتابة batch: **~9k run/ثانية** محلياً (5000 run في ~0.56s).
- evaluate: عشرات آلاف الأمثلة/ثانية للدوال الخفيفة، وتسارع خطي مع workers للدوال I/O (LLM).

## تثبيت

```bash
pip install -e .  # الحزمة تشمل pysmith* (انظر pyproject.toml)
# بلا اعتماديات إجبارية. اختياري للسرعة: pip install orjson
```

## استعمال مطابق لـ langsmith

```python
from pysmith import traceable, trace, tracing_context, Client, evaluate, exact_match

@traceable(name="qa", run_type="chain", tags=["v1"])
def answer(q: str) -> str:
    return "الإجابة: " + q

# context manager مثل langsmith.trace
with trace(name="step", inputs={"a": 1}) as run:
    out = answer("مرحبا")

# تعطيل مقطع كامل (تكلفة ~صفر)
with tracing_context(enabled=False):
    answer("بلا تتبع")

# datasets / examples / runs / feedback — نفس أسماء langsmith.Client
c = Client()  # تخزين محلي .pysmith/ + .pysmith_runs.jsonl
c.create_dataset("qa-v1", description="أسئلة عربية")
c.create_examples("qa-v1", [{"inputs": {"q": "مرحبا"}, "outputs": {"a": "مرحبا"}}])
print(c.list_datasets(), c.list_examples("qa-v1"))

res = evaluate(lambda inp: {"a": inp["q"]}, "qa-v1",
               evaluators=[exact_match], client=c, max_workers=8)
print(res.summary)  # {'n':..,'errors':..,'avg_latency':..,'p50_latency':..,'scores':{...}}

run_id = c.list_runs(limit=1)[-1]["id"]
c.create_feedback(run_id, "correct", score=1.0, comment="ممتاز")

# prompts hub محلي (مطابق create_prompt/get_prompt/list_prompts)
c.create_prompt("greet", "مرحبا {name}")
print(c.get_prompt("greet"))

# async
import asyncio
from pysmith import aevaluate
ares = asyncio.run(aevaluate(lambda i: {"a": i["q"]}, "qa-v1", evaluators=[exact_match]))
```

## التوافق مع langsmith (خريطة API)

- `Client`: `create/read/has/list/delete_dataset` (+ aliases `*_project`)،
  `create_example(s)` / `read/list/update/delete_example(s)`،
  `create_llm/chat_example`، `create_example_from_run`،
  `list/read/create_run`، `batch_ingest_runs` (=`multipart_ingest`)،
  `get_run_url`، `get_run_stats`، `create/list/read/delete_feedback`،
  `evaluate/aevaluate`، `create/get/list/pull/push_prompt`، `flush/close/info`.
- `run_helpers`: `traceable` (=`trace` كـ decorator)، `trace` (decorator + context manager)،
  `tracing_context`، `get_current_run_tree` (=`current_run`)، `get_tracing_context`،
  `set_tracing_parent`، `set_run_metadata`، `Run`/`RunTree`، `flush/is_enabled/set_enabled`.
- `evaluate`: `evaluate` / `aevaluate` / `evaluate_run` / `ExperimentResults` (+ `summary` مع p50/p95)،
  مقيّمات: `exact_match`، `contains`، `regex_match`، `score_string`.
- `schemas`: `Dataset`، `Example`، `Feedback`، ...

أي كود langsmith يعمل بتغيير سطر الاستيراد غالباً:
```python
# from langsmith import Client, traceable, evaluate
from pysmith import Client, traceable, evaluate
```

## التحكم (env)

- `PYSMITH_TRACING=false` — تعطيل شامل (إنتاج: صفر overhead).
- `PYSMITH_STORE=.pysmith_runs.jsonl` — مسار runs.
- `PYSMITH_BATCH=100` — حجم دفعة الكاتب الخلفي.
- `PYSMITH_SAMPLE=1.0` — نسبة التتبع (0.1 = تتبع 10% فقط).
- `PYSMITH_ENDPOINT=http://...` — إرسال best-effort غير حاجب (مهلة 2s) + تخزين محلي دائماً.
- `LANGSMITH_API_KEY` يُقرأ كـ fallback لـ `PYSMITH_API_KEY` (توافق).

## بنية الحزمة

- `pysmith/_utils.py` — `fast_dumps` (orjson/stdlib)، `new_id`، sampling.
- `pysmith/tracing.py` — `Run/__slots__` + كاتب batch + `traceable/trace/tracing_context`.
- `pysmith/client.py` — Client محلي (cache + كتابة ذرّية + tail فعّال).
- `pysmith/evaluate.py` — تقييم متوازٍ + async + مقاييس p50/p95.
- `pysmith/prompts.py` — hub محلي بالإصدارات.
- `pysmith/schemas.py` — dataclasses خفيفة.

## اختبارات و benchmark

```bash
python -m pytest tests/test_pysmith.py -q
PYTHONPATH=. python benchmarks/bench_pysmith.py
```
