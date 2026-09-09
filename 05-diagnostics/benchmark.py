"""
Benchmark — classical unit-root tests vs topological vs statistical features.

The question is not whether topological features beat TSFresh; measured, they do
not. It is what they add to the battery a practitioner actually uses. ADF and
KPSS answer *whether* a series is non-stationary, not *why*, and that second
question is where persistence has something to contribute.

Three feature families, compared alone and in every combination:

    classical    8   ADF and KPSS statistic + p-value, each with constant and
                     with constant+trend
    topology    24   sub_H0 + sup_H0 + H1, 8 scalars per diagram, signed-log
                     transformed
    tsfresh     24   the top-24 by mutual information, so the budget matches
                     topology; also reported at the full selected 100

Two tasks, because they are not the same question:

    binary       stationary vs not — where the classical tests are the real
                 baseline and are genuinely competent
    9-class      which cause — where they are not

Two transforms are applied to the topological features and both are deliberate:

    signed log   carl_f3 and carl_f4 carry p^4 terms. Even on z-normalised input
                 they span seven orders of magnitude, which leaves distance-based
                 models unusable (SVM-RBF measured 0.586 raw against 0.774
                 logged) while trees are indifferent. The transform is monotone,
                 so nothing is lost.
    z-normalise  applied upstream, in both extractors, so the comparison is about
                 features rather than preprocessing. See modes.sh.

Classical features are cached per mode, since ADF with autolag is the slow part.

Usage:
    python 05-diagnostics/benchmark.py --mode shape
    python 05-diagnostics/benchmark.py --mode shape --repeats 5 --out results/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------

def _load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_parquet(path / "features.parquet")
    L = pd.read_parquet(path / "labels.parquet")
    for d in (X, L):
        if "series_id" in d.columns:
            d.set_index("series_id", inplace=True)
    return X, L


def signed_log(A: np.ndarray) -> np.ndarray:
    return np.sign(A) * np.log1p(np.abs(A))


def clean(A: np.ndarray) -> np.ndarray:
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


def classical_features(x: np.ndarray) -> list[float]:
    """ADF and KPSS, each with constant and constant+trend: statistic and p-value."""
    from statsmodels.tsa.stattools import adfuller, kpss
    out: list[float] = []
    for reg in ("c", "ct"):
        try:
            r = adfuller(x, regression=reg, autolag="AIC")
            out += [r[0], r[1]]
        except Exception:
            out += [0.0, 1.0]
    for reg in ("c", "ct"):
        try:
            r = kpss(x, regression=reg, nlags="auto")
            out += [r[0], r[1]]
        except Exception:
            out += [0.0, 0.1]
    return out


CLASSICAL_NAMES = ["adf_c_stat", "adf_c_p", "adf_ct_stat", "adf_ct_p",
                   "kpss_c_stat", "kpss_c_p", "kpss_ct_stat", "kpss_ct_p"]


def get_classical(mode: str, order, cache_dir: Path) -> np.ndarray:
    cache = cache_dir / f"classical_{mode}_{len(order)}.npy"
    if cache.exists():
        C = np.load(cache)
        if len(C) == len(order):
            print(f"  classical: cached ({C.shape})")
            return C
    raw = pd.read_parquet(REPO / "data" / "raw" / mode / f"{mode}.parquet",
                          columns=["series_id", "time", "data"])
    series = {s: np.asarray(g.sort_values("time")["data"].tolist(), dtype=float)
              for s, g in raw.groupby("series_id")}
    print(f"  classical: computing ADF/KPSS for {len(order)} series...")
    C = np.array(Parallel(n_jobs=-1)(delayed(classical_features)(series[s]) for s in order))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, C)
    return C


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="shape")
    ap.add_argument("--repeats", type=int, default=3, help="CV repeats")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="directory for CSV output")
    ap.add_argument("--cache-dir", default=None, help="where to cache classical features")
    args = ap.parse_args()

    R = REPO / "data" / "features" / args.mode
    Xt, L = _load(R / "topological")
    Xa, _ = _load(R / "allfeatures")
    Xs, _ = _load(R / "selected" / "primary")
    idx = Xt.index.intersection(Xa.index).intersection(Xs.index).intersection(L.index)
    Xt, Xa, Xs, L = Xt.loc[idx], Xa.loc[idx], Xs.loc[idx], L.loc[idx]

    y9 = np.asarray(L["primary_category"].astype(str).tolist(), dtype=object)
    y2 = (y9 != "stationary").astype(int)
    classes = sorted(set(y9))

    print("=" * 78)
    print(f"Benchmark — mode={args.mode}  {len(idx)} series  {len(classes)} classes")
    print(f"CV: {args.folds}-fold x {args.repeats} repeats, seed {args.seed}")
    print("=" * 78)

    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO / "data" / "cache"
    C = get_classical(args.mode, list(idx), cache_dir)
    T = signed_log(clean(Xt.to_numpy(float)))
    A = clean(Xa.to_numpy(float))
    mi = mutual_info_classif(A, y9, random_state=args.seed)
    S24 = A[:, np.argsort(mi)[::-1][:24]]
    S100 = clean(Xs.to_numpy(float))
    print(f"  topology {T.shape[1]} | tsfresh pool {A.shape[1]} -> top24 | selected {S100.shape[1]}")

    SETS = {
        "classical (8)":            C,
        "topology (24)":            T,
        "tsfresh (24)":             S24,
        "tsfresh (100)":            S100,
        "classical+topology (32)":  np.hstack([C, T]),
        "classical+tsfresh (32)":   np.hstack([C, S24]),
        "tsfresh+topology (48)":    np.hstack([S24, T]),
        "all three (56)":           np.hstack([C, S24, T]),
        "tsfresh100+topology (124)": np.hstack([S100, T]),
    }

    rf = lambda: RandomForestClassifier(300, random_state=args.seed, n_jobs=-1)
    rcv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                  random_state=args.seed)

    rows, folds = [], {}
    print(f"\n  {'feature set':28s} {'binary':>16s} {'9-class':>16s}")
    print("  " + "-" * 62)
    for name, X in SETS.items():
        b = cross_val_score(rf(), X, y2, cv=rcv, n_jobs=1)
        m = cross_val_score(rf(), X, y9, cv=rcv, n_jobs=1)
        folds[name] = m
        rows.append(dict(feature_set=name, n_features=X.shape[1],
                         binary_mean=b.mean(), binary_std=b.std(),
                         multi_mean=m.mean(), multi_std=m.std()))
        print(f"  {name:28s} {b.mean():.3f} +/- {b.std():.3f}   {m.mean():.3f} +/- {m.std():.3f}")

    # Does topology add anything on top of each baseline?
    print("\n  What topology adds (9-class, paired over folds)")
    print("  " + "-" * 62)
    for base, combo in [("classical (8)", "classical+topology (32)"),
                        ("tsfresh (24)", "tsfresh+topology (48)"),
                        ("classical+tsfresh (32)", "all three (56)"),
                        ("tsfresh (100)", "tsfresh100+topology (124)")]:
        d = folds[combo] - folds[base]
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = "adds" if d.mean() > 2 * se else ("no gain" if d.mean() > -2 * se else "hurts")
        print(f"  {base:26s} -> +topology  {d.mean():+.4f} +/- {se:.4f}   {verdict}")
    print("  Note: CV folds overlap, so these standard errors are optimistic.")

    cv1 = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    per_class = {}
    for name in ("classical (8)", "topology (24)", "classical+topology (32)",
                 "tsfresh (24)", "all three (56)"):
        pred = cross_val_predict(rf(), SETS[name], y9, cv=cv1)
        per_class[name] = f1_score(y9, pred, average=None, labels=classes)
    F = pd.DataFrame(per_class, index=classes)
    print("\n  Per-class F1 (single 5-fold run)")
    print("  " + "-" * 62)
    print(F.round(3).to_string())

    # The flat model reports the hierarchy for free: P(not stationary) = 1 - P(stationary)
    print("\n  Non-stationarity score from the 9-class model, vs the binary task")
    print("  " + "-" * 62)
    k = classes.index("stationary")
    for name in ("classical (8)", "topology (24)", "classical+topology (32)", "tsfresh (24)"):
        P = cross_val_predict(rf(), SETS[name], y9, cv=cv1, method="predict_proba")
        print(f"  {name:28s} AUC = {roc_auc_score(y2, 1.0 - P[:, k]):.3f}")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / f"{args.mode}_summary.csv", index=False)
        F.to_csv(out / f"{args.mode}_per_class_f1.csv")
        print(f"\n  Saved -> {out}/{args.mode}_summary.csv, {args.mode}_per_class_f1.csv")


if __name__ == "__main__":
    main()
