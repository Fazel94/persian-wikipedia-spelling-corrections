"""Benchmark of `fadict.confidence`: how well `p_fix` separates real spelling fixes from the rest.

Out-of-fold predictions on the hand-labelled pairs (`annotation/pair_labels.tsv`), 5 folds
repeated 10 times with fixed seeds. Pairs sharing a corrected word stay in one fold. Each
labelled pair is weighted by the number of released pairs it stands for: its stratum (`high`,
else the `error_type`) has N pairs in the release and n labelled ones, so the weight is N / n and
the figures estimate the whole release. `spacing` has no labels and is left out.

Prints `METRIC name=value` lines (means over the repeats) and `ASI key=value` reference lines:
the same figures for the two hand-written rules, the spread of `cv_ap`, and accuracy per stratum.

    PYTHONPATH=. python scripts/bench_confidence.py
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from fadict import confidence

FOLDS = 5
REPEATS = 10
BOOTSTRAPS = 500
THRESHOLD = 0.5


def stratum(p: dict) -> str:
    return "high" if p["confidence"] == "high" else p["error_type"]


def prf(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn = w[y & pred].sum(), w[~y & pred].sum(), w[y & ~pred].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main() -> int:
    pairs = confidence.load_pairs()
    labels = confidence.read_labels()
    missing = [k for k in labels if k not in pairs]
    if missing:
        print(f"{len(missing)} labelled pairs are not in the release, e.g. {missing[0]}", file=sys.stderr)
        return 1
    keys = sorted(labels)
    rows = [pairs[k] for k in keys]
    y = np.array([labels[k] for k in keys])
    strata = np.array([stratum(p) for p in rows])
    population = Counter(stratum(p) for p in pairs.values())
    labelled = Counter(strata)
    w = np.array([population[s] / labelled[s] for s in strata])
    groups = np.array([p["right"] for p in rows])
    X = np.array(confidence.featurize(rows), dtype=float)

    runs = []
    oof = np.zeros((REPEATS, len(keys)))
    warnings.filterwarnings("ignore", message="The least populated class")
    for rep in range(REPEATS):
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=rep)
        for fit, held in folds.split(X, [f"{s}:{v}" for s, v in zip(strata, y)], groups):
            model = confidence.train(X[fit].tolist(), y[fit].tolist(), w[fit].tolist())
            oof[rep, held] = confidence.predict(model, X[held].tolist())
        p = oof[rep]
        runs.append((average_precision_score(y, p, sample_weight=w), *prf(y, p >= THRESHOLD, w),
                     roc_auc_score(y, p, sample_weight=w), np.average((p - y) ** 2, weights=w)))
    ap, precision, recall, f1, auc, brier = np.mean(runs, axis=0)
    print(f"METRIC cv_ap={ap:.4f}")
    print(f"METRIC cv_f1={f1:.4f}")
    print(f"METRIC cv_precision={precision:.4f}")
    print(f"METRIC cv_recall={recall:.4f}")
    print(f"METRIC cv_auc={auc:.4f}")
    print(f"METRIC cv_brier={brier:.4f}")

    # How far cv_ap moves when the labelled pairs are redrawn within each stratum.
    rng = np.random.default_rng(0)
    members = [np.flatnonzero(strata == s) for s in sorted(labelled)]
    p = oof.mean(axis=0)
    draws = []
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate([rng.choice(m, len(m)) for m in members])
        draws.append(average_precision_score(y[idx], p[idx], sample_weight=w[idx]))
    print(f"ASI cv_ap_bootstrap_sd={np.std(draws):.4f}")
    print(f"ASI cv_ap_repeat_sd={np.std([r[0] for r in runs]):.4f}")
    print(f"ASI labelled_pairs={len(keys)}")
    print(f"ASI effective_pairs={w.sum() ** 2 / (w ** 2).sum():.0f}")
    print(f"ASI fix_share={np.average(y, weights=w):.4f}")
    for name, rule in (("high", strata == "high"),
                       ("non_word", np.array([r["error_type"] == "non_word" for r in rows]))):
        rp, rr, rf = prf(y, rule, w)
        print(f"ASI rule_{name}_precision={rp:.4f}")
        print(f"ASI rule_{name}_recall={rr:.4f}")
        print(f"ASI rule_{name}_f1={rf:.4f}")
    for s in sorted(labelled):
        m = strata == s
        print(f"ASI accuracy_{s}={np.mean((oof[:, m] >= THRESHOLD) == y[m]):.4f}")
        print(f"ASI n_{s}={labelled[s]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
