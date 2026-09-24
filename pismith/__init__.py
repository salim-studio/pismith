# Copyright (c) 2026 salim-slimani — MIT License
"""pismith — the toolkit for developers, analysts & data scientists (local-first).

    # LLM observability (langsmith-compatible)
    from pismith import traceable, trace, tracing_context, Client, evaluate
    # Databases
    from pismith import Database, connect
    # Data (analysts)
    from pismith import DataTable, profile, clean, train_test_split
    # Files
    from pismith import load, save
    # ML experiments (lightweight MLflow alternative)
    from pismith import Experiments
    # ML/DL metrics
    from pismith import mae, rmse, r2_score, accuracy, f1, classification_report
    # ETL pipelines
    from pismith import Pipeline, step

Zero required dependencies. Every module auto-accelerates with
pandas/orjson/duckdb when available.
"""
from __future__ import annotations

__version__ = "0.3.0"

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

# --- databases + data + ML ---
from .db import Database, connect
from .data import (DataTable, correlation, drop_duplicates, fill_missing,
                   filter_rows, groupby_agg, infer_schema, profile, clean,
                   select, shuffle, sort_rows, train_test_split, validate)
from .io import load, save, load_csv, load_json, load_jsonl, load_sqlite
from .experiments import Experiments
from .metrics import (accuracy, bleu1, cer, classification_report, confusion_matrix,
                      f1, levenshtein, mae, mape, mse, precision_recall_f1,
                      r2_score, regression_report, rmse, rouge_l_f1,
                      silhouette_score, top_k_accuracy, roc_auc, wer,
                      make_metric_evaluator)
from .pipeline import Pipeline, Step, step

__all__ = ["__version__",
           # langsmith parity
           "traceable", "trace", "tracing_context", "Run", "RunTree",
           "current_run", "get_current_run_tree", "get_run_tree_context",
           "get_tracing_context", "set_tracing_parent", "set_run_metadata",
           "is_traceable_function", "ensure_traceable",
           "flush", "is_enabled", "set_enabled",
           "Client", "evaluate", "aevaluate", "evaluate_run",
           "ExperimentResults", "exact_match", "contains",
           "regex_match", "score_string",
           "prompts", "Dataset", "Example", "Feedback",
           # db
           "Database", "connect",
           # data
           "DataTable", "profile", "clean", "fill_missing", "drop_duplicates",
           "filter_rows", "select", "sort_rows", "groupby_agg", "validate",
           "shuffle", "train_test_split", "correlation", "infer_schema",
           # io
           "load", "save", "load_csv", "load_json", "load_jsonl", "load_sqlite",
           # experiments
           "Experiments",
           # metrics
           "mae", "mse", "rmse", "mape", "r2_score", "regression_report",
           "accuracy", "confusion_matrix", "precision_recall_f1", "f1",
           "top_k_accuracy", "roc_auc", "classification_report",
           "silhouette_score", "levenshtein", "cer", "wer", "bleu1", "rouge_l_f1",
           "make_metric_evaluator",
           # pipeline
           "Pipeline", "Step", "step"]
