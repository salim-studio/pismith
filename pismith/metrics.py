"""Metrics — pure-Python ML/DL metrics (zero dependencies).

Covers: regression + classification + clustering + text/LLM + ranking.
Every function takes lists and returns a float/dict — and plugs into
evaluate() as an evaluator.

    from pismith import mae, rmse, r2_score, accuracy, f1, confusion_matrix
    mae([3, -0.5, 2], [2.5, 0.0, 2])   # 0.33..
    accuracy([0,1,1], [0,0,1])          # 0.66..
"""
from __future__ import annotations

import math
from collections import Counter


def _to_list(x):
    try:
        return list(x)
    except Exception:
        return [x]


# ---------- regression ----------
def mae(y_true, y_pred) -> float:
    a, b = _to_list(y_true), _to_list(y_pred)
    assert len(a) == len(b) and a, "length mismatch/empty"
    return round(sum(abs(x - y) for x, y in zip(a, b)) / len(a), 6)


def mse(y_true, y_pred) -> float:
    a, b = _to_list(y_true), _to_list(y_pred)
    assert len(a) == len(b) and a
    return round(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a), 6)


def rmse(y_true, y_pred) -> float:
    return round(math.sqrt(mse(y_true, y_pred)), 6)


def mape(y_true, y_pred) -> float:
    a, b = _to_list(y_true), _to_list(y_pred)
    vals = [abs((x - y) / x) for x, y in zip(a, b) if x != 0]
    return round(100 * sum(vals) / len(vals), 4) if vals else 0.0


def r2_score(y_true, y_pred) -> float:
    a, b = _to_list(y_true), _to_list(y_pred)
    m = sum(a) / len(a)
    ss_tot = sum((x - m) ** 2 for x in a)
    ss_res = sum((x - y) ** 2 for x, y in zip(a, b))
    return round(1 - ss_res / ss_tot, 6) if ss_tot else 0.0


def regression_report(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred), "n": len(_to_list(y_true))}


# ---------- classification ----------
def accuracy(y_true, y_pred) -> float:
    a, b = _to_list(y_true), _to_list(y_pred)
    return round(sum(1 for x, y in zip(a, b) if x == y) / max(1, len(a)), 6)


def confusion_matrix(y_true, y_pred, labels: list | None = None) -> dict:
    a, b = _to_list(y_true), _to_list(y_pred)
    labels = labels or sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            m[idx[x]][idx[y]] += 1
    return {"labels": labels, "matrix": m}


def precision_recall_f1(y_true, y_pred, average: str = "binary", pos_label=1) -> dict:
    """average: binary/micro/macro — pure Python."""
    a, b = _to_list(y_true), _to_list(y_pred)
    labels = sorted(set(a) | set(b))
    if average == "binary":
        tp = sum(1 for x, y in zip(a, b) if x == pos_label and y == pos_label)
        fp = sum(1 for x, y in zip(a, b) if x != pos_label and y == pos_label)
        fn = sum(1 for x, y in zip(a, b) if x == pos_label and y != pos_label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f, 6)}
    if average == "micro":
        tp = sum(1 for x, y in zip(a, b) if x == y)
        p = r = f = tp / max(1, len(a))
        return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f, 6)}
    # macro
    ps, rs, fs = [], [], []
    for lab in labels:
        s = precision_recall_f1(a, b, average="binary", pos_label=lab)
        ps.append(s["precision"])
        rs.append(s["recall"])
        fs.append(s["f1"])
    n = max(1, len(labels))
    return {"precision": round(sum(ps) / n, 6), "recall": round(sum(rs) / n, 6),
            "f1": round(sum(fs) / n, 6)}


def f1(y_true, y_pred, **kw) -> float:
    return precision_recall_f1(y_true, y_pred, **kw)["f1"]


def top_k_accuracy(y_true, y_topk: list[list], k: int = 3) -> float:
    a = _to_list(y_true)
    return round(sum(1 for x, preds in zip(a, y_topk) if x in list(preds)[:k]) / max(1, len(a)), 6)


def roc_auc(y_true, y_score) -> float:
    """AUC via Mann-Whitney (binary only)."""
    scores = _to_list(y_score)
    labels = _to_list(y_true)
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y != 1]
    if not pos or not neg:
        return 0.0
    wins = sum(1 for p in pos for n_ in neg if p > n_) + 0.5 * sum(1 for p in pos for n_ in neg if p == n_)
    return round(wins / (len(pos) * len(neg)), 6)


def classification_report(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": accuracy(y_true, y_pred),
            "macro": precision_recall_f1(y_true, y_pred, average="macro"),
            "micro": precision_recall_f1(y_true, y_pred, average="micro"),
            "confusion": cm, "n": len(_to_list(y_true))}


# ---------- clustering ----------
def silhouette_samples(X: list[list[float]], labels: list) -> list[float]:
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    out = []
    for i, (x, lab) in enumerate(zip(X, labels)):
        same = [j for j, l in enumerate(labels) if l == lab and j != i]
        other_labels = set(labels) - {lab}
        a = sum(dist(x, X[j]) for j in same) / len(same) if same else 0.0
        b_vals = []
        for ol in other_labels:
            js = [j for j, l in enumerate(labels) if l == ol]
            b_vals.append(sum(dist(x, X[j]) for j in js) / len(js))
        b = min(b_vals) if b_vals else 0.0
        out.append(round((b - a) / max(a, b), 4) if max(a, b) else 0.0)
    return out


def silhouette_score(X, labels) -> float:
    s = silhouette_samples(list(X), list(labels))
    return round(sum(s) / len(s), 6) if s else 0.0


# ---------- text / LLM ----------
def levenshtein(a, b) -> int:
    """Edit distance — works on strings or token/word lists."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate — handy for OCR and morphologically rich languages."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return round(levenshtein(reference, hypothesis) / max(1, len(reference)), 6)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: Levenshtein distance at the word level."""
    r = str(reference).split()
    h = str(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    return round(levenshtein(r, h) / max(1, len(r)), 6)


def bleu1(reference: str, hypothesis: str) -> float:
    ref = Counter(str(reference).split())
    hyp = str(hypothesis).split()
    if not hyp:
        return 0.0
    overlap = sum(min(ref[w], Counter(hyp)[w]) for w in set(hyp))
    return round(overlap / len(hyp), 6)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    r, h = str(reference).split(), str(hypothesis).split()
    if not r or not h:
        return 0.0
    # LCS
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if r[i] == h[j] else max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    p = lcs / len(h)
    rec = lcs / len(r)
    return round(2 * p * rec / (p + rec), 6) if p + rec else 0.0


# ---------- ready-made evaluators for evaluate() ----------
def make_metric_evaluator(metric_fn, name: str | None = None, pred_key: str = "prediction",
                          true_key: str = "target"):
    def _ev(example: dict, outputs) -> dict:
        outs = outputs if isinstance(outputs, dict) else {pred_key: outputs}
        refs = example.get("outputs") or {}
        pred = outs.get(pred_key, outs.get("output", next(iter(outs.values()), None) if outs else None))
        true = refs.get(true_key, refs.get("output", next(iter(refs.values()), None) if refs else None))
        try:
            return {(name or metric_fn.__name__): float(metric_fn([true], [pred]))}
        except Exception:
            try:
                return {(name or metric_fn.__name__): float(metric_fn(true, pred))}
            except Exception as e:
                return {(name or metric_fn.__name__ + "_error"): str(e)[:200]}
    _ev.__name__ = name or metric_fn.__name__
    return _ev


__all__ = ["mae", "mse", "rmse", "mape", "r2_score", "regression_report",
           "accuracy", "confusion_matrix", "precision_recall_f1", "f1",
           "top_k_accuracy", "roc_auc", "classification_report",
           "silhouette_samples", "silhouette_score",
           "levenshtein", "cer", "wer", "bleu1", "rouge_l_f1",
           "make_metric_evaluator"]
