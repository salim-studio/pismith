"""Schemas — مطابقة لـ langsmith.schemas (نسخة خفيفة بـ dataclasses + slots)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Dataset:
    id: str
    name: str
    description: str = ""
    created_at: float = 0.0
    example_count: int = 0


@dataclass(slots=True)
class Example:
    id: str
    dataset_id: str
    inputs: dict
    outputs: dict | None = None
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(slots=True)
class ExampleCreate:
    inputs: dict
    outputs: dict | None = None
    metadata: dict | None = None
    id: str | None = None


@dataclass(slots=True)
class RunSchema:
    id: str
    name: str
    run_type: str = "chain"
    inputs: dict | None = None
    outputs: dict | None = None
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    parent_id: str | None = None
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class Feedback:
    id: str
    run_id: str
    key: str
    score: float | None = None
    comment: str | None = None
    created_at: float = 0.0


@dataclass(slots=True)
class PromptVersion:
    name: str
    template: str
    commit: str = "v1"
    variables: list = field(default_factory=list)


def to_dict(obj) -> dict:
    try:
        return asdict(obj)
    except Exception:
        return dict(obj.__dict__) if hasattr(obj, "__dict__") else {}


__all__ = ["Dataset", "Example", "ExampleCreate", "RunSchema", "Feedback",
           "PromptVersion", "to_dict"]
