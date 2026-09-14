"""Tests — DB + Data + IO + Experiments + Metrics + Pipeline (zero dependencies)."""
import os
import tempfile

from pismith import (Database, DataTable, Experiments, Pipeline, accuracy, cer,
                     classification_report, clean, f1, groupby_agg, load, mae,
                     profile, r2_score, rmse, save, step, train_test_split,
                     validate)


def test_db_sqlite_memory():
    db = Database.memory()
    db.execute("CREATE TABLE t (a TEXT, b REAL)")
    assert db.insert_many("t", [{"a": "x", "b": 1.0}, {"a": "y", "b": 2.0}]) == 2
    rows = db.query("SELECT * FROM t ORDER BY b")
    assert [r["a"] for r in rows] == ["x", "y"]
    assert db.count("t") == 2
    assert "t" in db.list_tables()
    assert {c["name"] for c in db.schema("t")} >= {"a", "b"}
    assert db.query_value("SELECT SUM(b) AS s FROM t") == 3.0
    df_like = db.query_df("SELECT * FROM t")
    assert len(df_like) == 2
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        db.to_csv("t", p)
        assert db.from_csv(p, "t2") == 2
        assert db.count("t2") == 2
    db.close()


def test_data_table_chain():
    rows = [{"city": " Oran ", "amount": 10}, {"city": "Oran", "amount": None},
            {"city": "", "amount": 5}, {"city": "Oran", "amount": 10}]
    t = DataTable(rows).clean().fill_missing({"amount": 0}).drop_duplicates()
    assert len(t) <= 4
    p = t.profile()
    assert p["n"] == len(t) and "quality" in p
    g = groupby_agg(t.to_list(), "city", {"amount": "sum"})
    assert isinstance(g, list)
    v = validate(t.to_list(), {"amount": {"min": 0, "not_null": True}})
    assert v["valid"]
    tr, te = train_test_split(t.to_list(), test_size=0.5, seed=1)
    assert len(tr) + len(te) == len(t)
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "c.csv")
        t.to_csv(p1)
        assert len(load(p1)) == len(t)
        p2 = os.path.join(d, "c.json")
        save(t.to_list(), p2)
        assert load(p2)
    assert "DataTable" in repr(t)


def test_experiments_local():
    with tempfile.TemporaryDirectory() as d:
        xp = Experiments(base_dir=os.path.join(d, "xp"))
        xp.create_experiment("e1")
        with xp.run("e1", params={"lr": 0.1}) as r:
            r.log_metric("acc", 0.9)
        with xp.run("e1", params={"lr": 0.01}) as r:
            r.log_metrics({"acc": 0.95})
        runs = xp.list_runs("e1")
        assert len(runs) == 2
        assert xp.best_run("e1", metric="acc")["metrics"]["acc"] == 0.95
        assert len(xp.compare("e1")) == 2
        xp.to_csv("e1", os.path.join(d, "runs.csv"))


def test_metrics_pure_python():
    assert mae([1, 2], [1, 3]) == 0.5
    assert rmse([0, 0], [3, 4]) == 3.535534
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([0, 1, 1], [0, 0, 1]) == 0.666667
    assert 0.0 <= f1([0, 1, 1], [0, 0, 1]) <= 1.0
    rep = classification_report([0, 1], [0, 1])
    assert rep["accuracy"] == 1.0
    assert cer("مرحبا", "مرحبا") == 0.0
    assert cer("abc", "abd") > 0


def test_pipeline_with_cache():
    calls = {"n": 0}

    @step(cache=True)
    def s1(ctx):
        calls["n"] += 1
        return [{"x": 1}]

    @step
    def s2(ctx, rows):
        return [{**r, "y": 2} for r in rows]

    pipe = Pipeline([s1, s2], name="p")
    assert pipe.run() == [{"x": 1, "y": 2}]
    assert pipe.run() == [{"x": 1, "y": 2}]
    assert calls["n"] == 1  # cache ضرب
    assert pipe.summary()["steps"] == 2


def test_clean_profile_edge():
    assert profile([])["n"] == 0
    assert clean([{"a": "  x  "}])[0]["a"] == "x"
