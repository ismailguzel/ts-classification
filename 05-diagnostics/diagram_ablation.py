"""
Diagram Ablation — where do topological features work, and what does each
homology dimension actually capture?

This is the study's central instrument. Reporting one accuracy number for
"topology (18 features)" hides the finding: it cannot say whether the signal
came from sublevel H0, superlevel H0, or H1. So instead of one lumped model we
fit the same classifier on each diagram family separately and read the per-class
F1 side by side.

Feature subsets compared (raw 24 topological features, 8 per diagram — not the
MI-selected subset, so every family keeps the same budget and the comparison is
symmetric):

    sub_H0        8   sublevel filtration  — valleys, mean level, trend
    sup_H0        8   superlevel filtration — peaks, point anomalies
    H1            8   Takens + Vietoris-Rips — loops, i.e. periodicity
    H0            16  sub_H0 + sup_H0
    topology      24  all three
    tsfresh      100  MI-selected statistical baseline
    tsfresh+topo 124  both

The expectation the two-study split was built on: H1 should carry Study B and
contribute almost nothing to Study A. This script is what confirms or refutes
that.

Usage:
    python diagram_ablation.py --mode shape
    python diagram_ablation.py --mode season-anomaly --classifier xgboost
    python diagram_ablation.py --mode shape --subsets sub_H0,H1,tsfresh
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent

# Diagram family -> regex over the raw topological feature names
FAMILIES = {
    "sub_H0": r"^topo__sub_H0__",
    "sup_H0": r"^topo__sup_H0__",
    "H1":     r"^topo__H1__",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a features/labels pair, indexed by series_id."""
    for cand in (path / "primary", path):
        f, l = cand / "features.parquet", cand / "labels.parquet"
        if f.exists() and l.exists():
            X = pd.read_parquet(f)
            y = pd.read_parquet(l)
            for df in (X, y):
                if "series_id" in df.columns:
                    df.set_index("series_id", inplace=True)
            return X, y
    raise FileNotFoundError(f"No features.parquet + labels.parquet under {path}")


def load_mode(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (topological 24, tsfresh 100, class labels) aligned on series_id."""
    feat_root = REPO / "data" / "features" / mode
    X_topo, lab = _load(feat_root / "topological")
    X_tsf, _ = _load(feat_root / "selected")

    idx = X_topo.index.intersection(X_tsf.index).intersection(lab.index)
    if len(idx) == 0:
        raise ValueError(f"No series_id overlap between the feature sets for '{mode}'")

    y = lab.loc[idx, "primary_category"].astype(str)
    return X_topo.loc[idx], X_tsf.loc[idx], y


def build_subsets(X_topo: pd.DataFrame, X_tsf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Assemble the feature subsets to compare."""
    cols = {name: [c for c in X_topo.columns if re.match(pat, c)]
            for name, pat in FAMILIES.items()}

    missing = [n for n, c in cols.items() if not c]
    if missing:
        print(f"  warning: no columns matched for {missing} — "
              f"available: {sorted(X_topo.columns)[:4]}...", file=sys.stderr)

    subsets: dict[str, pd.DataFrame] = {}
    for name, c in cols.items():
        if c:
            subsets[name] = X_topo[c]
    if cols["sub_H0"] and cols["sup_H0"]:
        subsets["H0"] = X_topo[cols["sub_H0"] + cols["sup_H0"]]
    subsets["topology"] = X_topo
    subsets["tsfresh"] = X_tsf
    subsets["tsfresh+topo"] = X_tsf.join(X_topo, how="inner")
    return subsets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_classifier(name: str, seed: int, n_jobs: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs,
                             tree_method="hist", verbosity=0)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, random_seed=seed, verbose=0,
                                  thread_count=n_jobs if n_jobs > 0 else -1)
    raise ValueError(f"Unknown classifier: {name}")


def evaluate(X: pd.DataFrame, y_codes: np.ndarray, classes: list[str],
             clf_name: str, seed: int, test_size: float, n_jobs: int) -> tuple[float, np.ndarray]:
    """Fit on a stratified split and return (accuracy, per-class F1)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X.to_numpy(dtype=float), y_codes,
        test_size=test_size, random_state=seed, stratify=y_codes)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0))
    Xte = scaler.transform(np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0))

    clf = make_classifier(clf_name, seed, n_jobs)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
    return acc, f1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, title: str, floatfmt: str = "{:.3f}") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    w = max(len(str(i)) for i in df.index) + 2
    header = " " * w + "".join(f"{c:>14s}" for c in df.columns)
    print(header)
    for idx, row in df.iterrows():
        cells = "".join(f"{floatfmt.format(v):>14s}" for v in row)
        print(f"{str(idx):<{w}s}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    help="pipeline mode: shape | season-structure | season-anomaly")
    ap.add_argument("--classifier", default="rf", choices=["rf", "xgboost", "catboost"])
    ap.add_argument("--subsets", default=None,
                    help="comma-separated subset names to restrict to")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None, help="directory for CSV output")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Diagram ablation — mode={args.mode}  classifier={args.classifier}  seed={args.seed}")
    print("=" * 78)

    X_topo, X_tsf, y = load_mode(args.mode)
    classes = sorted(y.unique())
    y_codes = y.map({c: i for i, c in enumerate(classes)}).to_numpy()

    print(f"Series          : {len(y)}")
    print(f"Classes         : {len(classes)}  {classes}")
    print(f"Topo features   : {X_topo.shape[1]} (raw, pre-selection)")
    print(f"TSFresh features: {X_tsf.shape[1]} (MI-selected)")

    subsets = build_subsets(X_topo, X_tsf)
    if args.subsets:
        want = [s.strip() for s in args.subsets.split(",")]
        unknown = [s for s in want if s not in subsets]
        if unknown:
            raise SystemExit(f"Unknown subset(s): {unknown}. Available: {list(subsets)}")
        subsets = {k: subsets[k] for k in want}

    acc_rows, f1_rows = {}, {}
    for name, X in subsets.items():
        acc, f1 = evaluate(X, y_codes, classes, args.classifier,
                           args.seed, args.test_size, args.n_jobs)
        acc_rows[name] = {"n_features": X.shape[1], "accuracy": acc}
        f1_rows[name] = f1
        print(f"  {name:<14s} {X.shape[1]:>4d} features   accuracy = {acc:.3f}")

    acc_df = pd.DataFrame(acc_rows).T
    f1_df = pd.DataFrame(f1_rows, index=classes)

    print_table(acc_df[["accuracy"]].T.rename(index={"accuracy": "accuracy"}),
                "Accuracy by feature subset")
    print_table(f1_df, "Per-class F1 by feature subset")

    # Which subset each class is best served by, and by how much over the runner-up.
    topo_only = [c for c in ("sub_H0", "sup_H0", "H1") if c in f1_df.columns]
    if topo_only:
        best = f1_df[topo_only].idxmax(axis=1)
        margin = f1_df[topo_only].max(axis=1) - f1_df[topo_only].apply(
            lambda r: r.nlargest(2).iloc[-1] if len(r) > 1 else 0.0, axis=1)
        print("\nWhich single diagram family serves each class best")
        print("-" * 51)
        for cls in f1_df.index:
            print(f"  {cls:<24s} {best[cls]:<8s} F1={f1_df.loc[cls, best[cls]]:.3f} "
                  f"(+{margin[cls]:.3f} over next)")

    if "tsfresh" in f1_df.columns and "topology" in f1_df.columns:
        delta = (f1_df["topology"] - f1_df["tsfresh"]).sort_values(ascending=False)
        print("\nTopology (24) minus TSFresh (100), per class")
        print("-" * 44)
        for cls, d in delta.items():
            mark = "topology" if d > 0.02 else ("tsfresh" if d < -0.02 else "tie")
            print(f"  {cls:<24s} {d:+.3f}   {mark}")

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_{args.classifier}"
    acc_df.to_csv(out_dir / f"{stem}_accuracy.csv")
    f1_df.to_csv(out_dir / f"{stem}_per_class_f1.csv")
    print(f"\nSaved -> {out_dir / (stem + '_accuracy.csv')}")
    print(f"Saved -> {out_dir / (stem + '_per_class_f1.csv')}")


if __name__ == "__main__":
    main()
