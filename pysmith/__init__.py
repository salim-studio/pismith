"""pysmith — بديل langsmith الأسرع (تتبع محلي أولاً + تقييم متوازٍ).

مطابقة الواجهة:
    from pysmith import traceable, trace, tracing_context, Client, evaluate
    # نفس أسماء langsmith: Client.create_dataset/list_examples/create_feedback/...
    # run_helpers: traceable, trace, tracing_context, get_current_run_tree

لماذا أسرع؟
- `PYSMITH_TRACING=false` → تكلفة ~صفر (bool واحد).
- كاتب خلفية batch واحد + orjson إن وُجد (بدل POST متزامن لكل run).
- `__slots__` + تسلسل محدود الحجم + sampling عبر `PYSMITH_SAMPLE`.
- evaluate متوازٍ (threads) + aevaluate (asyncio).

متغيرات البيئة:
- PYSMITH_TRACING=true/false
- PYSMITH_STORE=.pysmith_runs.jsonl
- PYSMITH_BATCH=100  (حجم الدفعة)
- PYSMITH_SAMPLE=1.0 (نسبة التتبع 0..1)
- PYSMITH_ENDPOINT=http://... (إرسال best-effort غير حاجب، اختياري)
"""
from __future__ import annotations

__version__ = "0.2.0"

from .tracing import (Run, RunTree, current_run, ensure_traceable, flush,
                      get_current_run_tree, get_run_tree_context,
                      get_tracing_context, is_enabled, is_traceable_function,
                      set_enabled, set_run_metadata, set_tracing_parent, trace,
                      traceable, tracing_context)
from .client import Client
from .evaluate import (ExperimentResults, aevaluate, contains, evaluate,
                       evaluate_run, exact_match, regex_match, score_string)
from . import prompts as prompts
from .schemas import Dataset, Example, Feedback

__all__ = ["__version__",
           "traceable", "trace", "tracing_context", "Run", "RunTree",
           "current_run", "get_current_run_tree", "get_run_tree_context",
           "get_tracing_context", "set_tracing_parent", "set_run_metadata",
           "is_traceable_function", "ensure_traceable",
           "flush", "is_enabled", "set_enabled",
           "Client", "evaluate", "aevaluate", "evaluate_run",
           "ExperimentResults", "exact_match", "contains",
           "regex_match", "score_string",
           "prompts", "Dataset", "Example", "Feedback"]
