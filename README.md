<p align="center">
  <img src="assets/banner.svg" alt="pismith — forge raw data into products" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/salim-studio/pismith"><img src="https://img.shields.io/github/stars/salim-studio/pismith?style=social" alt="GitHub stars"/></a>
  <a href="https://github.com/salim-studio/pismith/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-amber" alt="MIT license"/></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.9-3776AB?logo=python&logoColor=white" alt="Python >=3.9"/>
  <img src="https://img.shields.io/badge/deps-zero_required-22c55e" alt="Zero required dependencies"/>
  <img src="https://img.shields.io/github/last-commit/salim-studio/pismith" alt="Last commit"/>
  <a href="https://github.com/salim-studio/pismith/issues"><img src="https://img.shields.io/github/issues/salim-studio/pismith" alt="Open issues"/></a>
</p>

# pismith — forge raw data into products 🔥

> **One local-first toolkit for developers, analysts & data scientists:**
> databases · data analysis · ML/DL experiments · LLM observability.
> **Zero required dependencies** — every module auto-accelerates with `pandas` / `orjson` / `duckdb` when present.

Like a blacksmith's forge, **pismith** (`π` + `smith`) takes raw material — rows, files, model runs, LLM traces — and hammers it into something finished.

## Install

```bash
pip install pismith
# fastest JSON engine (optional): pip install orjson
# pandas interop (optional):      pip install pandas pyarrow openpyxl

# before the PyPI release lands, install straight from GitHub:
pip install git+https://github.com/salim-studio/pismith.git
# contributors:
git clone https://github.com/salim-studio/pismith.git && cd pismith && pip install -e ".[all]"
```

## 30-second tour

```python
from pismith import Database, DataTable, Experiments, traceable, evaluate, exact_match

# 1) Databases — one API for SQLite/Postgres/MySQL/DuckDB/Mongo
db = Database("sqlite:///sales.db")          # or Database.memory() for tests
db.insert_many("sales", [{"city": "Oran", "amount": 120.5}])
print(db.query("SELECT city, SUM(amount) AS total FROM sales GROUP BY city"))

# 2) Data analysis — no pandas required
t = DataTable(db.query("SELECT * FROM sales")).clean().fill_missing({"amount": 0})
print(t.describe())                          # stats + quality score per column
train, test = t.split(test_size=0.2, seed=7)

# 3) ML experiments — local-first MLflow alternative (no server)
xp = Experiments()
with xp.run("pricing", params={"lr": 0.01}) as r:
    r.log_metric("rmse", 3.21)
print(xp.best_run("pricing", metric="rmse", mode="min"))

# 4) LLM observability — langsmith-compatible, faster
@traceable(name="qa", tags=["v1"])
def answer(q: str) -> str:
    return "Answer: " + q
answer("hello")
```

## Why pismith?

| You are… | Instead of learning 5 libraries… | You write… |
|---|---|---|
| Backend dev | `sqlite3` + `psycopg2` + `pymongo` + hand SQL | `Database("sqlite:///app.db")` — one API, always `list[dict]` |
| Data analyst | heavy `pandas` + slow profilers | `DataTable(rows).clean().profile()` — works **without pandas** |
| Data scientist | scattered cleaning snippets | `fill_missing`, `validate`, `groupby_agg`, `train_test_split` built in |
| ML engineer | `mlflow` server + setup | `Experiments()` — local, `best_run`, `compare`, `to_csv` |
| DL/LLM engineer | cloud tracing + API keys + per-run POST | `traceable` + `evaluate` — local-first, ~µs overhead when off |
| Everyone | a different snippet per file type | `load("any.csv|xlsx|parquet|jsonl|db")` / `save(rows, "out.parquet")` |

## Modules

| Module | What it does | Zero-dep? |
|---|---|---|
| `pismith.db` — `Database`, `connect` | Unified SQLite/Postgres/MySQL/DuckDB/Mongo access, `query`, `insert_many`, `schema`, CSV/JSON import-export | ✅ (drivers optional) |
| `pismith.data` — `DataTable`, `profile`, `clean`, `validate`, `groupby_agg`, `train_test_split` | Chainable analysis on `list[dict]`, HTML reports, pandas bridge | ✅ |
| `pismith.io` — `load`, `save` | Auto-detect CSV/JSON/JSONL/Parquet/Excel/SQLite/URL | ✅ (format libs optional) |
| `pismith.experiments` — `Experiments` | Params/metrics/artifacts tracking, `best_run`, `compare`, CSV export | ✅ |
| `pismith.metrics` — `mae`, `rmse`, `r2_score`, `accuracy`, `f1`, `roc_auc`, `cer`, `wer`, `bleu1`, `rouge_l_f1`… | Pure-Python ML/DL/NLP metrics | ✅ |
| `pismith.pipeline` — `Pipeline`, `@step` | Traceable ETL with per-step cache | ✅ |
| `pismith.tracing` / `client` / `evaluate` / `prompts` | langsmith-compatible runs, datasets, parallel eval, prompt hub | ✅ |

## LLM tracing (langsmith-compatible)

Most langsmith code runs by changing one import:

```python
# from langsmith import Client, traceable, evaluate
from pismith import Client, traceable, trace, tracing_context, evaluate, aevaluate, exact_match

c = Client()  # local store: .pismith/ + .pismith_runs.jsonl
c.create_dataset("qa-v1")
c.create_examples("qa-v1", [{"inputs": {"q": "hi"}, "outputs": {"a": "hi"}}])
res = evaluate(lambda i: {"a": i["q"]}, "qa-v1", evaluators=[exact_match], client=c)
print(res.summary)  # {'n':…, 'errors':…, 'avg_latency':…, 'p50_latency':…, 'scores':{…}}

with tracing_context(enabled=False):   # ~zero-cost section in production
    answer("not traced")
```

## Configuration (env)

| Variable | Default | Effect |
|---|---|---|
| `PISMITH_TRACING` | `true` | `false` disables all tracing (~1 µs/call) |
| `PISMITH_STORE` | `.pismith_runs.jsonl` | Run log path |
| `PISMITH_BATCH` | `100` | Background writer batch size |
| `PISMITH_SAMPLE` | `1.0` | Trace sampling rate (`0.1` = 10%) |
| `PISMITH_ENDPOINT` | — | Best-effort non-blocking mirror + always-local store |
| `PISMITH_API_KEY` | — | Optional auth header value |

> **Migrating from `pysmith`?** Rename the import (`pysmith` → `pismith`) and you're done:
> `PISMITH_*` vars take precedence, but legacy `PYSMITH_*` values, `__pysmith_traceable__`
> markers, and `LANGSMITH_API_KEY` are still honored. Defaults moved to `.pismith*` paths.

## CLI

```bash
python -m pismith profile data.csv
python -m pismith query sales.db "SELECT city, SUM(amount) FROM sales GROUP BY city"
python -m pismith version
```

## Benchmarks

```bash
python -m pytest tests/ -q
python benchmarks/bench_pismith.py
python examples/examples_db_ml.py
```

Reference numbers (Windows, CPython): tracing **disabled ≈ 760k calls/s (~1.3 µs)**;
**enabled + batch write ≈ 9k runs/s** locally; `evaluate` scales linearly with workers on I/O-bound targets.

## Project layout

```
pismith/            ← the library (db, data, io, experiments, metrics, pipeline, tracing, …)
tests/              ← pytest suite (runs with zero third-party deps)
benchmarks/         ← bench_pismith.py
examples/           ← end-to-end demo
assets/             ← logo.svg, banner.svg (brand kit)
```

## Roadmap

- [ ] PyPI release + version badge
- [ ] `pismith serve` — tiny local dashboard for runs & experiments
- [ ] Data connectors: BigQuery / Snowflake / S3 read helpers
- [ ] More metrics: NDCG, MAP, calibration curves
- [ ] Recipes: `examples/` notebooks for churn, OCR quality, RAG eval

## Contributing

Issues and PRs are welcome — especially new **metrics**, **connectors**, and **examples**.
Please keep the core dependency-free: optional integrations must degrade gracefully with a clear `pip install` hint.

## License

MIT © 2026 [salim-studio](https://github.com/salim-studio) — see [LICENSE](LICENSE).
Formerly `pysmith`; renamed to `pismith` at v0.3.0 (history preserved).
