"""Database — a unified database layer (local-first, zero required dependencies).

Why you'll like it:
- One API for every database: SQLite (built-in) + Postgres/MySQL/DuckDB/Mongo
  (optional, only if their drivers are installed).
- Always returns `list[dict]` (analyst-friendly) + `query_df()` returns pandas
  when available.
- Integrated with pismith tracing: every query is traced automatically.
- One-call CSV/JSON export and fast schema inspection.

Examples:
    from pismith import Database
    db = Database("sqlite:///sales.db")   # or Database.memory() or Database("sales.db")
    db.execute("CREATE TABLE IF NOT EXISTS sales (id INTEGER, city TEXT, amount REAL)")
    db.insert_many("sales", [{"city": "Oran", "amount": 120.5}])
    rows = db.query("SELECT city, SUM(amount) FROM sales GROUP BY city")
    print(db.schema("sales"), db.list_tables())
    db.to_csv("SELECT * FROM sales", "sales.csv")

    # Postgres (needs psycopg2): Database("postgresql://user:pw@localhost/db")
    # DuckDB (needs duckdb):     Database("duckdb:///analytics.duckdb")
    # Mongo (needs pymongo):     Database("mongodb://localhost:27017/mydb.mycol")
"""
from __future__ import annotations

import csv
import os
import re
import sqlite3
import threading
import time

from ._utils import fast_dumps, new_id


def _parse_url(url: str) -> dict:
    """Parse sqlite:///path, sqlite:///:memory:, postgresql://, mysql://, duckdb://, mongodb://"""
    if url in (":memory:", "memory", "sqlite:///:memory:"):
        return {"kind": "sqlite", "path": ":memory:"}
    if url.endswith((".db", ".sqlite", ".sqlite3")) or url == ".pismith_data.db":
        return {"kind": "sqlite", "path": url}
    m = re.match(r"^(\w+)(?:\+\w+)?://(.*)$", url or "")
    if not m:
        # treat it as a sqlite file path
        return {"kind": "sqlite", "path": url or ":memory:"}
    scheme, rest = m.group(1).lower(), m.group(2)
    if scheme == "sqlite":
        path = rest.lstrip("/")
        if not path or path == ":memory:":
            path = ":memory:"
        return {"kind": "sqlite", "path": path}
    if scheme in ("postgres", "postgresql"):
        return {"kind": "postgres", "dsn": url}
    if scheme in ("mysql", "mariadb"):
        return {"kind": "mysql", "dsn": url}
    if scheme in ("duckdb", "duck"):
        path = rest.lstrip("/") or ":memory:"
        return {"kind": "duckdb", "path": path}
    if scheme in ("mongo", "mongodb"):
        # mongodb://host/db.collection
        return {"kind": "mongo", "dsn": url}
    return {"kind": "sqlite", "path": url}


class Database:
    """One unified interface. Thread-safe for simple read/write (a single lock)."""

    def __init__(self, url: str = ":memory:", trace: bool = True):
        self.url = url
        self.info = _parse_url(url)
        self.kind = self.info["kind"]
        self.trace = trace
        self._lock = threading.RLock()
        self._local = threading.local()
        if self.kind == "sqlite" and self.info["path"] != ":memory:":
            d = os.path.dirname(os.path.abspath(self.info["path"]))
            if d:
                os.makedirs(d, exist_ok=True)
        if self.kind == "mongo":
            self._mongo_col = self._connect_mongo()

    # ---------- constructors ----------
    @classmethod
    def memory(cls, **kw) -> "Database":
        return cls(":memory:", **kw)

    @classmethod
    def file(cls, path: str, **kw) -> "Database":
        return cls(path, **kw)

    # ---------- low-level ----------
    def _sqlite_conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "sqlite", None)
        if c is None:
            c = sqlite3.connect(self.info["path"], check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._local.sqlite = c
        return c

    def _connect_mongo(self):
        try:
            import pymongo  # type: ignore
        except Exception as e:
            raise ImportError("MongoDB support needs: pip install pymongo") from e
        dsn = self.info["dsn"]
        # expected form: .../dbname.collection — grab the trailing part
        m = re.match(r"^(mongodb://[^/]+)/([^.]+)(?:\.(.+))?$", dsn)
        if not m:
            raise ValueError("Use mongodb://host:port/dbname.collection")
        client = pymongo.MongoClient("mongodb://" + m.group(1).split("://", 1)[1])
        return client[m.group(2)][m.group(3) or "docs"]

    def _external_conn(self):
        kind = self.kind
        if kind == "postgres":
            try:
                import psycopg2  # type: ignore
                import psycopg2.extras  # type: ignore
            except Exception as e:
                raise ImportError("Postgres support needs: pip install psycopg2-binary") from e
            return psycopg2.connect(self.info["dsn"])
        if kind == "mysql":
            try:
                import pymysql  # type: ignore
            except Exception as e:
                raise ImportError("MySQL support needs: pip install pymysql") from e
            from urllib.parse import urlparse
            u = urlparse(self.info["dsn"])
            return pymysql.connect(host=u.hostname or "localhost", port=u.port or 3306,
                                   user=u.username or "root", password=u.password or "",
                                   database=(u.path or "/").lstrip("/") or None,
                                   cursorclass=pymysql.cursors.DictCursor)
        if kind == "duckdb":
            try:
                import duckdb  # type: ignore
            except Exception as e:
                raise ImportError("DuckDB support needs: pip install duckdb") from e
            return duckdb.connect(self.info["path"])
        raise ValueError(f"Unsupported kind: {kind}")

    def _maybe_trace(self, name: str, sql: str):
        if not self.trace:
            return None
        try:
            from . import tracing as _t
            if not _t.is_enabled():
                return None
            run = _t.Run(name, {"db": self.kind, "sql": sql[:500]}, "tool",
                         metadata={"db_url": self.url[:80]})
            return run
        except Exception:
            return None

    def _finish_trace(self, run, outputs=None, error=None):
        if run is None:
            return
        try:
            from . import tracing as _t
            import time as _time
            run.outputs = outputs
            run.error = error
            run.end = _time.time()
            _t._log_run(run)
        except Exception:
            pass

    # ---------- core API ----------
    def execute(self, sql: str, params=None) -> dict:
        """Run INSERT/CREATE/... — returns {rowcount, lastrowid, time_ms}."""
        run = self._maybe_trace("db.execute", sql)
        t0 = time.time()
        try:
            with self._lock:
                if self.kind == "sqlite":
                    cur = self._sqlite_conn().execute(sql, params or [])
                    self._sqlite_conn().commit()
                    out = {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid,
                           "time_ms": round((time.time() - t0) * 1000, 2)}
                elif self.kind == "mongo":
                    raise ValueError("MongoDB has no SQL — use insert_docs/find_docs")
                else:
                    conn = self._external_conn()
                    try:
                        cur = conn.cursor()
                        cur.execute(sql, params or ())
                        conn.commit()
                        out = {"rowcount": getattr(cur, "rowcount", -1),
                               "time_ms": round((time.time() - t0) * 1000, 2)}
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
            self._finish_trace(run, {"ok": True, **out})
            return out
        except Exception as e:
            self._finish_trace(run, error=e)
            raise

    def query(self, sql: str, params=None, limit: int = 100000) -> list[dict]:
        """SELECT → list[dict] (works with sqlite/postgres/mysql/duckdb)."""
        run = self._maybe_trace("db.query", sql)
        try:
            with self._lock:
                if self.kind == "mongo":
                    raise ValueError("MongoDB: use find_docs(filter) instead")
                if self.kind == "sqlite":
                    cur = self._sqlite_conn().execute(sql, params or [])
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = [dict(zip(cols, r)) for r in cur.fetchmany(limit)]
                else:
                    conn = self._external_conn()
                    try:
                        cur = conn.cursor()
                        cur.execute(sql, params or ())
                        raw = cur.fetchmany(limit)
                        if raw and isinstance(raw[0], dict):
                            rows = [dict(r) for r in raw]
                        else:
                            cols = [d[0] for d in (cur.description or [])]
                            rows = [dict(zip(cols, r)) for r in raw]
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
            self._finish_trace(run, {"n": len(rows)})
            return rows
        except Exception as e:
            self._finish_trace(run, error=e)
            raise

    def query_df(self, sql: str, params=None, limit: int = 100000):
        """Like query, but returns a pandas.DataFrame when pandas is installed, else list[dict]."""
        rows = self.query(sql, params, limit)
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(rows)
        except Exception:
            return rows

    def query_value(self, sql: str, params=None):
        rows = self.query(sql, params, limit=1)
        if not rows:
            return None
        return next(iter(rows[0].values()), None)

    # ---------- analyst helpers ----------
    def list_tables(self) -> list[str]:
        if self.kind == "sqlite":
            rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return [r["name"] for r in rows]
        if self.kind == "duckdb":
            rows = self.query("SHOW TABLES")
            return [next(iter(r.values())) for r in rows]
        if self.kind == "mongo":
            return [self._mongo_col.name]
        rows = self.query("SELECT table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema')")
        return [next(iter(r.values())) for r in rows]

    def schema(self, table: str) -> list[dict]:
        """One entry per column: {name, type} — fast on every backend."""
        if self.kind == "sqlite":
            return [{"name": r["name"], "type": r["type"]}
                    for r in self.query(f'PRAGMA table_info("{table}")')]
        if self.kind == "mongo":
            doc = self._mongo_col.find_one() or {}
            return [{"name": k, "type": type(v).__name__} for k, v in doc.items()]
        try:
            rows = self.query(f"SELECT * FROM {table} LIMIT 1")
        except Exception:
            rows = []
        if rows:
            return [{"name": k, "type": type(v).__name__} for k, v in rows[0].items()]
        return []

    def count(self, table: str, where: str = "", params=None) -> int:
        sql = f'SELECT COUNT(*) AS n FROM "{table}"' + (f" WHERE {where}" if where else "")
        return int(self.query_value(sql, params) or 0)

    def insert_many(self, table: str, rows: list[dict], batch: int = 500) -> int:
        """Fast batched inserts — auto-creates the table when missing (sqlite)."""
        if not rows:
            return 0
        if self.kind == "mongo":
            with self._lock:
                self._mongo_col.insert_many(rows)
            return len(rows)
        cols = list(rows[0].keys())
        if self.kind == "sqlite":
            with self._lock:
                conn = self._sqlite_conn()
                # إنشاء تلقائي بأنواع مُستنتجة
                sample = rows[0]
                def _t(v):
                    if isinstance(v, int):
                        return "INTEGER"
                    if isinstance(v, float):
                        return "REAL"
                    return "TEXT"
                coldefs = ", ".join(f'"{c}" {_t(sample.get(c))}' for c in cols)
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({coldefs})')
                ph = ", ".join(["?"] * len(cols))
                names = ", ".join(f'"{c}"' for c in cols)
                n = 0
                for i in range(0, len(rows), batch):
                    chunk = rows[i:i + batch]
                    conn.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({ph})',
                                     [[r.get(c) for c in cols] for r in chunk])
                    n += len(chunk)
                conn.commit()
                return n
        # خارجي: executemany عادي
        ph = ", ".join(["%s"] * len(cols))
        names = ", ".join(cols)
        total = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            conn = self._external_conn()
            try:
                cur = conn.cursor()
                cur.executemany(f"INSERT INTO {table} ({names}) VALUES ({ph})",
                                [[r.get(c) for c in cols] for r in chunk])
                conn.commit()
                total += len(chunk)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        return total

    # ----- Mongo helpers -----
    def insert_docs(self, docs: list[dict]) -> int:
        if self.kind != "mongo":
            # local fallback: store in a docs table
            return self.insert_many("docs", docs)
        with self._lock:
            self._mongo_col.insert_many(docs)
        return len(docs)

    def find_docs(self, filt: dict | None = None, limit: int = 1000) -> list[dict]:
        if self.kind != "mongo":
            return self.query("SELECT * FROM docs LIMIT ?", (limit,))
        out = []
        for d in self._mongo_col.find(filt or {}).limit(limit):
            d["id"] = str(d.pop("_id", d.get("id", new_id())))
            out.append(d)
        return out

    # ----- import / export -----
    def to_csv(self, sql_or_table: str, path: str) -> str:
        rows = self.query(sql_or_table) if " " in sql_or_table.strip() else self.query(f'SELECT * FROM "{sql_or_table}"')
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["_empty"])
            w.writeheader()
            w.writerows(rows)
        return path

    def from_csv(self, path: str, table: str, batch: int = 500) -> int:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return self.insert_many(table, rows, batch)

    def to_json(self, sql_or_table: str, path: str) -> str:
        rows = self.query(sql_or_table) if " " in sql_or_table.strip() else self.query(f'SELECT * FROM "{sql_or_table}"')
        with open(path, "w", encoding="utf-8") as f:
            f.write(fast_dumps(rows))
        return path

    def close(self):
        try:
            c = getattr(self._local, "sqlite", None)
            if c is not None:
                c.commit()
                c.close()
                self._local.sqlite = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def connect(url: str = ":memory:", **kw) -> Database:
    """Shortcut: connect("sales.db") / connect("sqlite:///x.db") / connect("postgresql://...")."""
    return Database(url, **kw)


__all__ = ["Database", "connect"]
