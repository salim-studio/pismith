"""Data — analyst & data-scientist toolkit (works without pandas, faster with it).

Philosophy: `list[dict]` (rows) first — the simplest structure everyone
understands — with pandas/sklearn interop when available, never required.

    from pismith import DataTable, profile, clean, train_test_split

    rows = [{"city": "Oran", "amount": 120}, {"city": "Algiers", "amount": None}]
    print(profile(rows))                 # instant stats + quality score
    t = DataTable(rows).clean().fill_missing({"amount": 0}).drop_duplicates()
    train, test = t.split(test_size=0.2, seed=7)   # or train_test_split(rows, ...)
    print(t.describe(), t.to_csv("clean.csv"))

With pandas (optional):
    df = t.to_pandas()          # needs pandas
    t2 = DataTable.from_pandas(df)
"""
from __future__ import annotations

import csv
import math
import os
import random
import statistics
from collections import Counter

from ._utils import fast_dumps


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if _is_num(x)]
    return round(statistics.fmean(xs), 4) if xs else None


def infer_schema(rows: list[dict], sample: int = 200) -> dict:
    """Infer each column's type: integer/float/string/bool/mixed/unknown."""
    if not rows:
        return {}
    cols: dict[str, set] = {}
    for r in rows[:sample]:
        for k, v in r.items():
            s = cols.setdefault(k, set())
            if v is None or v == "":
                s.add("null")
            elif isinstance(v, bool):
                s.add("bool")
            elif isinstance(v, int):
                s.add("integer")
            elif isinstance(v, float):
                s.add("float")
            else:
                s.add("string")
    out = {}
    for k, s in cols.items():
        s2 = s - {"null"}
        if not s2:
            out[k] = "unknown"
        elif s2 == {"integer"}:
            out[k] = "integer"
        elif s2 <= {"integer", "float"}:
            out[k] = "float"
        elif s2 == {"bool"}:
            out[k] = "bool"
        elif s2 == {"string"}:
            out[k] = "string"
        else:
            out[k] = "mixed"
    return out


def profile(rows: list[dict], top_n: int = 5) -> dict:
    """Instant quality/stats report — a concise pandas-profiling analogue."""
    n = len(rows)
    if not n:
        return {"n": 0, "columns": {}, "quality": {"duplicates": 0, "score": 100}}
    cols = sorted({k for r in rows for k in r.keys()})
    seen: set = set()
    dups = 0
    columns: dict = {}
    total_missing = 0
    for c in cols:
        vals = [r.get(c) for r in rows]
        missing = sum(1 for v in vals if v is None or v == "")
        total_missing += missing
        uniq = len({fast_dumps(v) for v in vals})
        nums = [v for v in vals if _is_num(v)]
        col: dict = {"missing": missing, "missing_pct": round(100 * missing / n, 2),
                     "unique": uniq, "unique_pct": round(100 * uniq / n, 2)}
        if nums:
            col.update({"mean": _mean(nums), "min": min(nums), "max": max(nums),
                        "std": round(statistics.pstdev(nums), 4) if len(nums) > 1 else 0.0})
        else:
            strs = [str(v) for v in vals if v not in (None, "")]
            if strs:
                col["top"] = [{"value": v, "count": k} for v, k in Counter(strs).most_common(top_n)]
                col["min_len"] = min(len(s) for s in strs)
                col["max_len"] = max(len(s) for s in strs)
        columns[c] = col
    for r in rows:
        key = fast_dumps({k: r.get(k) for k in cols})
        if key in seen:
            dups += 1
        seen.add(key)
    miss_pct = total_missing / max(1, n * len(cols))
    score = round(100 * (1 - 0.6 * miss_pct - 0.4 * (dups / max(1, n))), 1)
    return {"n": n, "n_columns": len(cols), "columns": columns,
            "quality": {"duplicates": dups, "missing_cells": total_missing,
                        "score": max(0.0, score)}}


def clean(rows: list[dict], *, strip_strings: bool = True, drop_empty_rows: bool = True,
          drop_columns: list | None = None, rename: dict | None = None) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        if rename:
            for a, b in rename.items():
                if a in d:
                    d[b] = d.pop(a)
        if drop_columns:
            for c in drop_columns:
                d.pop(c, None)
        if strip_strings:
            for k, v in list(d.items()):
                if isinstance(v, str):
                    v2 = v.strip()
                    d[k] = v2
        if drop_empty_rows and all(v in (None, "") for v in d.values()):
            continue
        out.append(d)
    return out


def fill_missing(rows: list[dict], fill: dict | str = "mean") -> list[dict]:
    """fill: {col: value} or 'mean'/'median'/'mode' for numeric columns."""
    if isinstance(fill, dict):
        return [{**r, **{k: (v if r.get(k) not in (None, "") else v2)
                         for k, v2 in fill.items() for v in [r.get(k)]}} for r in rows]
    nums_cols = [c for c in ({k for r in rows for k in r} or [])]
    stat: dict = {}
    for c in nums_cols:
        nums = sorted(v for r in rows for v in [r.get(c)] if _is_num(v))
        if not nums:
            continue
        if fill == "mean":
            stat[c] = statistics.fmean(nums)
        elif fill == "median":
            stat[c] = statistics.median(nums)
        elif fill == "mode":
            stat[c] = Counter(nums).most_common(1)[0][0]
    return [{k: (stat[k] if (v in (None, "") and k in stat) else v) for k, v in r.items()} for r in rows]


def drop_duplicates(rows: list[dict], subset: list | None = None) -> list[dict]:
    seen: set = set()
    out = []
    for r in rows:
        key = fast_dumps({k: r.get(k) for k in (subset or sorted(r.keys()))})
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def filter_rows(rows: list[dict], where=None, **equals) -> list[dict]:
    if where is None and equals:
        return [r for r in rows if all(r.get(k) == v for k, v in equals.items())]
    if callable(where):
        return [r for r in rows if where(r)]
    return list(rows)


def select(rows: list[dict], columns: list[str]) -> list[dict]:
    return [{c: r.get(c) for c in columns} for r in rows]


def sort_rows(rows: list[dict], by: str, reverse: bool = False) -> list[dict]:
    return sorted(rows, key=lambda r: (r.get(by) is None, r.get(by)), reverse=reverse)


def groupby_agg(rows: list[dict], by: str | list, agg: dict) -> list[dict]:
    """agg example: {"amount": "sum"} — supports sum/mean/min/max/count/median/nunique."""
    keys = [by] if isinstance(by, str) else list(by)
    groups: dict = {}
    for r in rows:
        k = tuple(r.get(c) for c in keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, g in groups.items():
        d = dict(zip(keys, k))
        for col, fn in agg.items():
            vals = [r.get(col) for r in g if r.get(col) is not None]
            nums = [v for v in vals if _is_num(v)]
            if fn == "count":
                d[f"{col}_count"] = len(vals)
            elif fn == "nunique":
                d[f"{col}_nunique"] = len({fast_dumps(v) for v in vals})
            elif fn == "sum":
                d[f"{col}_sum"] = round(sum(nums), 4) if nums else 0
            elif fn == "mean":
                d[f"{col}_mean"] = _mean(nums)
            elif fn == "min":
                d[f"{col}_min"] = min(nums) if nums else None
            elif fn == "max":
                d[f"{col}_max"] = max(nums) if nums else None
            elif fn == "median":
                d[f"{col}_median"] = statistics.median(nums) if nums else None
            else:
                raise ValueError(f"Unknown agg: {fn}")
        d["_n"] = len(g)
        out.append(d)
    return out


def validate(rows: list[dict], rules: dict) -> dict:
    """rules: {col: callable(v)->bool | {"min":..,"max":..,"not_null":True,"allowed":[...]}}."""
    errors: list[dict] = []
    for i, r in enumerate(rows):
        for col, rule in rules.items():
            v = r.get(col)
            ok = True
            if callable(rule):
                try:
                    ok = bool(rule(v))
                except Exception:
                    ok = False
            elif isinstance(rule, dict):
                if rule.get("not_null") and v in (None, ""):
                    ok = False
                if "min" in rule and _is_num(v) and v < rule["min"]:
                    ok = False
                if "max" in rule and _is_num(v) and v > rule["max"]:
                    ok = False
                if "allowed" in rule and v not in rule["allowed"]:
                    ok = False
            if not ok:
                errors.append({"row": i, "column": col, "value": v})
    return {"n": len(rows), "errors": len(errors), "valid": len(errors) == 0,
            "pass_rate": round(1 - len(errors) / max(1, len(rows)), 4), "details": errors[:50]}


def shuffle(rows: list[dict], seed: int = 42) -> list[dict]:
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def train_test_split(rows: list[dict], test_size: float = 0.2, seed: int = 42):
    s = shuffle(rows, seed)
    k = int(len(s) * (1 - test_size))
    return s[:k], s[k:]


def correlation(rows: list[dict], col_x: str, col_y: str) -> float | None:
    pairs = [(r[col_x], r[col_y]) for r in rows
             if _is_num(r.get(col_x)) and _is_num(r.get(col_y))]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den else 0.0


class DataTable:
    """A lightweight chainable table — a mini-pandas for beginners + a bridge to it."""

    def __init__(self, rows: list[dict] | None = None):
        self.rows: list[dict] = [dict(r) for r in (rows or [])]

    # -- constructors --
    @classmethod
    def from_csv(cls, path: str, **kw) -> "DataTable":
        with open(path, encoding=kw.get("encoding", "utf-8-sig")) as f:
            return cls(list(csv.DictReader(f)))

    @classmethod
    def from_json(cls, path: str) -> "DataTable":
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("rows", data.get("data", [data]))
        return cls(data)

    @classmethod
    def from_pandas(cls, df) -> "DataTable":
        return cls(df.to_dict(orient="records"))

    @classmethod
    def from_db(cls, db, sql: str, **kw) -> "DataTable":
        return cls(db.query(sql, **kw))

    # -- verbs (chainable) --
    def pipe(self, fn, *a, **k) -> "DataTable":
        return DataTable(fn(self.rows, *a, **k))

    def clean(self, **kw) -> "DataTable":
        return DataTable(clean(self.rows, **kw))

    def fill_missing(self, fill="mean") -> "DataTable":
        return DataTable(fill_missing(self.rows, fill))

    def drop_duplicates(self, subset=None) -> "DataTable":
        return DataTable(drop_duplicates(self.rows, subset))

    def filter(self, where=None, **kw) -> "DataTable":
        return DataTable(filter_rows(self.rows, where, **kw))

    def select(self, columns: list[str]) -> "DataTable":
        return DataTable(select(self.rows, columns))

    def sort(self, by: str, reverse: bool = False) -> "DataTable":
        return DataTable(sort_rows(self.rows, by, reverse))

    def groupby(self, by, agg: dict) -> "DataTable":
        return DataTable(groupby_agg(self.rows, by, agg))

    def sample(self, n: int = 5, seed: int = 42) -> "DataTable":
        return DataTable(shuffle(self.rows, seed)[:n])

    def split(self, test_size: float = 0.2, seed: int = 42):
        tr, te = train_test_split(self.rows, test_size, seed)
        return DataTable(tr), DataTable(te)

    # -- stats --
    def profile(self, **kw) -> dict:
        return profile(self.rows, **kw)

    def describe(self) -> dict:
        return profile(self.rows)

    def validate(self, rules: dict) -> dict:
        return validate(self.rows, rules)

    def shape(self) -> tuple:
        cols = {k for r in self.rows for k in r}
        return (len(self.rows), len(cols))

    def columns(self) -> list[str]:
        return sorted({k for r in self.rows for k in r})

    # -- export --
    def to_list(self) -> list[dict]:
        return [dict(r) for r in self.rows]

    def to_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            cols = self.columns() or ["_empty"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c, "") for c in cols})
        return path

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fast_dumps(self.rows))
        return path

    def to_pandas(self):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError("pandas support needs: pip install pandas") from e
        return pd.DataFrame(self.rows)

    def to_db(self, db, table: str) -> int:
        return db.insert_many(table, self.rows)

    def report_html(self, title: str = "pismith data report") -> str:
        """A small HTML report for quick analysis (save it and open in a browser)."""
        p = profile(self.rows)
        rows_html = "".join(
            f"<tr><td>{c}</td><td>{v.get('missing',0)}</td><td>{v.get('unique',0)}</td>"
            f"<td>{v.get('mean','')}</td><td>{v.get('min','')}</td><td>{v.get('max','')}</td></tr>"
            for c, v in p.get("columns", {}).items())
        return (f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>"
                f"<h1>{title}</h1><p>n={p.get('n')} — quality={p.get('quality',{}).get('score')}</p>"
                f"<table border=1><tr><th>col</th><th>missing</th><th>unique</th>"
                f"<th>mean</th><th>min</th><th>max</th></tr>{rows_html}</table></body></html>")

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        r, c = self.shape()
        return f"DataTable(rows={r}, cols={c}, columns={self.columns()[:6]})"


__all__ = ["DataTable", "profile", "clean", "fill_missing", "drop_duplicates",
           "filter_rows", "select", "sort_rows", "groupby_agg", "validate",
           "shuffle", "train_test_split", "correlation", "infer_schema"]
