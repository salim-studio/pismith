"""IO — smart load/save for any source (CSV/JSON/JSONL/Parquet/Excel/SQLite/URL).

    from pismith import load, save
    rows = load("sales.csv")          # always list[dict] (uniform)
    save(rows, "clean.parquet")       # format picked from the extension
    df = load("data.parquet", as_df=True)  # pandas when installed

Zero required dependencies: CSV/JSON/JSONL/SQLite come from the stdlib,
everything else is optional (pandas/pyarrow/openpyxl/requests) with clear messages.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3


def _ext(path: str) -> str:
    return os.path.splitext(path.split("?")[0].lower())[1]


def load_csv(path: str, encoding: str = "utf-8-sig", limit: int = 0) -> list[dict]:
    with open(path, encoding=encoding) as f:
        rdr = csv.DictReader(f)
        if limit and limit > 0:
            return [dict(r) for _, r in zip(range(limit), rdr)]
        return [dict(r) for r in rdr]


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(r) if isinstance(r, dict) else {"value": r} for r in data]
    if isinstance(data, dict):
        for k in ("rows", "data", "records", "examples"):
            if isinstance(data.get(k), list):
                return load_json_value(data[k])
    return [dict(data)]


def load_json_value(data: list) -> list[dict]:
    return [dict(r) if isinstance(r, dict) else {"value": r} for r in data]


def load_jsonl(path: str, limit: int = 0) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    v = json.loads(line)
                    out.append(dict(v) if isinstance(v, dict) else {"value": v})
                except Exception:
                    continue
    return out


def load_sqlite(path_or_db: str, sql: str = "SELECT * FROM sqlite_master", **kw) -> list[dict]:
    # path_or_db: "file.db" or "file.db::SELECT * FROM t"
    if "::" in path_or_db:
        path_or_db, sql = path_or_db.split("::", 1)
    conn = sqlite3.connect(path_or_db)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, kw.get("params") or [])
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def load(path: str, *, as_df: bool = False, limit: int = 0, sql: str | None = None,
         sheet: str | int = 0, encoding: str = "utf-8-sig") -> list[dict]:
    """Load any file/URL into a list[dict]."""
    if path.startswith(("http://", "https://")):
        return _load_url(path, as_df=as_df, limit=limit)
    e = _ext(path)
    if e == ".csv":
        rows = load_csv(path, encoding, limit)
    elif e in (".json",):
        rows = load_json(path)
        rows = rows[:limit] if limit else rows
    elif e == ".jsonl":
        rows = load_jsonl(path, limit)
    elif e in (".db", ".sqlite", ".sqlite3"):
        if sql:
            rows = load_sqlite(path, sql)
        else:
            tables = load_sqlite(path, "SELECT name FROM sqlite_master WHERE type='table'")
            first = next((r.get("name") for r in tables if r.get("name")), None)
            rows = load_sqlite(f"{path}::SELECT * FROM \"{first}\"") if first else []
    elif e in (".parquet", ".pq"):
        try:
            import pandas as pd  # type: ignore
        except Exception as ex:
            raise ImportError("Parquet support needs: pip install pandas pyarrow") from ex
        df = pd.read_parquet(path)
        return df if as_df else df.to_dict(orient="records")
    elif e in (".xlsx", ".xls"):
        try:
            import pandas as pd  # type: ignore
        except Exception as ex:
            raise ImportError("Excel support needs: pip install pandas openpyxl") from ex
        df = pd.read_excel(path, sheet_name=sheet)
        if not isinstance(df, type(df)):
            pass
        try:
            import pandas as _pd  # type: ignore
            if isinstance(df, _pd.DataFrame):
                pass
            else:  # dict of sheets
                df = list(df.values())[0]
        except Exception:
            pass
        return df if as_df else df.to_dict(orient="records")
    elif e in (".txt", ".tsv"):
        sep = "\t" if e == ".tsv" else None
        with open(path, encoding=encoding) as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                rdr = csv.DictReader(f, dialect=dialect)
            except Exception:
                rdr = csv.DictReader(f, delimiter=sep or ",")
            rows = [dict(r) for r in rdr]
            rows = rows[:limit] if limit else rows
    else:
        # try csv, then jsonl, then json
        try:
            rows = load_csv(path, encoding, limit)
        except Exception:
            try:
                rows = load_jsonl(path, limit)
            except Exception:
                rows = load_json(path)
    if as_df:
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(rows)
        except Exception as ex:
            raise ImportError("as_df needs: pip install pandas") from ex
    return rows


def _load_url(url: str, as_df: bool = False, limit: int = 0):
    e = _ext(url)
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read()
    except Exception as ex:
        # try requests when available
        try:
            import requests  # type: ignore
            raw = requests.get(url, timeout=15).content
        except Exception:
            raise RuntimeError(f"Could not download URL: {url}: {ex}") from ex
    import tempfile
    suffix = e or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(raw)
        tmp = t.name
    try:
        return load(tmp, as_df=as_df, limit=limit)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def save_csv(rows: list[dict], path: str) -> str:
    from .data import DataTable
    return DataTable(rows).to_csv(path)


def save_json(rows: list[dict], path: str, indent: int = 2) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=indent, default=str)
    return path


def save_jsonl(rows: list[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def save(rows, path: str, **kw) -> str:
    """Save rows (list[dict] or DataFrame) using the file extension."""
    e = _ext(path)
    try:
        import pandas as _pd  # type: ignore
        is_df = isinstance(rows, _pd.DataFrame)
    except Exception:
        is_df = False
    if is_df:
        if e == ".csv":
            rows.to_csv(path, index=False)
            return path
        if e == ".json":
            rows.to_json(path, orient="records", force_ascii=False, indent=2)
            return path
        if e == ".jsonl":
            rows.to_json(path, orient="records", lines=True, force_ascii=False)
            return path
        if e in (".parquet", ".pq"):
            rows.to_parquet(path, index=False)
            return path
        if e in (".xlsx",):
            rows.to_excel(path, index=False)
            return path
        rows = rows.to_dict(orient="records")
    if e == ".csv":
        return save_csv(rows, path)
    if e == ".json":
        return save_json(rows, path)
    if e == ".jsonl":
        return save_jsonl(rows, path)
    if e in (".parquet", ".pq", ".xlsx", ".xls"):
        try:
            import pandas as pd  # type: ignore
        except Exception as ex:
            raise ImportError("Saving this format needs: pip install pandas pyarrow openpyxl") from ex
        df = pd.DataFrame(rows)
        if e in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
        else:
            df.to_excel(path, index=False)
        return path
    # default: csv
    return save_csv(rows, path)


__all__ = ["load", "save", "load_csv", "load_json", "load_jsonl", "load_sqlite",
           "save_csv", "save_json", "save_jsonl"]
