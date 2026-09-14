"""End-to-end demo: DB + Data + Experiments + Metrics + Pipeline (zero dependencies)."""
import os
import tempfile

from pismith import (Database, DataTable, Experiments, Pipeline, accuracy, clean,
                     profile, regression_report, save, load, step, train_test_split,
                     classification_report)


def main():
    tmp = tempfile.mkdtemp(prefix="pismith_demo_")
    db = Database.memory()
    db.execute("CREATE TABLE sales (city TEXT, amount REAL)")
    db.insert_many("sales", [
        {"city": " Oran ", "amount": 120.5},
        {"city": "Algiers", "amount": 90.0},
        {"city": "Oran", "amount": None},
        {"city": "Oran", "amount": 120.5},
    ])
    print("tables:", db.list_tables(), "count:", db.count("sales"))

    rows = db.query("SELECT * FROM sales")
    t = DataTable(rows).clean().fill_missing({"amount": 0}).drop_duplicates()
    print("profile quality:", t.profile()["quality"])
    print("groupby:", t.groupby("city", {"amount": "sum"}).to_list())
    train, test = t.split(test_size=0.25, seed=7)
    print("split:", len(train), len(test))

    csv_path = os.path.join(tmp, "clean.csv")
    t.to_csv(csv_path)
    assert load(csv_path)
    print("csv ok:", csv_path)

    xp = Experiments(base_dir=os.path.join(tmp, "xp"))
    for lr in [0.1, 0.01]:
        with xp.run("demo", params={"lr": lr}) as r:
            r.log_metric("accuracy", 0.8 + lr)
    print("best:", xp.best_run("demo", metric="accuracy")["metrics"])
    print("regression:", regression_report([1, 2, 3], [1.1, 1.9, 3.2]))
    print("classification:", classification_report([0, 1, 1], [0, 0, 1])["macro"])
    assert abs(accuracy([0, 1], [0, 1]) - 1.0) < 1e-9

    @step
    def extract(ctx):
        return ctx["db"].query("SELECT * FROM sales")

    @step(cache=True)
    def transform(ctx, rows):
        return clean(rows)

    pipe = Pipeline([extract, transform], context={"db": db})
    out = pipe.run()
    print("pipeline:", pipe.summary(), "n:", len(out))
    print("OK — demo finished:", tmp)


if __name__ == "__main__":
    main()
