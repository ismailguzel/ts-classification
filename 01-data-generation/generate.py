"""
39-Class Time Series Dataset via BeTiSe
===========================================

Reads ``full-dataset-config.json`` (same shape as
``examples/configs/classification_config.json`` in the BeTiSe repo) and writes
one merged, shuffled parquet of labeled series.

Pipeline (see BeTiSe ``USAGE.md`` §6–7 and ``examples/05_classification_dataset.py``):

1. Load JSON: ``output_root``, ``output_name``, ``fixed_length``, ``random_seed``,
   and the ``classes`` map.
2. For each class and each scenario, call ``betise.config.load_config`` with a
   ``dataset`` block: ``base_series``, ``num_series``, ``length_range``,
   ``random_seed``, and merged ``features``.
3. Call ``generate_dataframe(cfg)`` (in-memory; no per-batch write unless you use
   ``betise.run``).

``primary_category`` is set to the JSON class key so your coarse label matches
the configured groups; BeTiSe’s default metadata would otherwise follow only the
last-applied feature family. ``sub_category`` keeps the fine-grained BeTiSe
composite label from the generation context.

Run (from this directory, with the project env active)::

    python generate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from betise import generate_dataframe
from betise.config import load_config

# ── Configuration ─────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="Generate time series dataset with betise")
_parser.add_argument("--config", type=str, default="full-dataset-config.json",
                     help="Config file name inside 01-data-generation/ (default: full-dataset-config.json)")
_args = _parser.parse_args()

CONFIG_FILE = Path(__file__).parent / _args.config

with open(CONFIG_FILE, encoding="utf-8") as f:
    CLASS_CFG = json.load(f)

OUTPUT_ROOT = CLASS_CFG["output_root"]
OUTPUT_NAME = CLASS_CFG["output_name"]
FIXED_LENGTH = CLASS_CFG["fixed_length"]
RANDOM_SEED = CLASS_CFG["random_seed"]
CLASSES = CLASS_CFG["classes"]

ALL_FEATURES_OFF = {
    "linear_trend": {"enabled": False},
    "quadratic_trend": {"enabled": False},
    "cubic_trend": {"enabled": False},
    "exponential_trend": {"enabled": False},
    "arch": {"enabled": False},
    "garch": {"enabled": False},
    "egarch": {"enabled": False},
    "aparch": {"enabled": False},
    "single_seasonality": {"enabled": False},
    "multiple_seasonality": {"enabled": False},
    "sarma": {"enabled": False},
    "sarima": {"enabled": False},
    "mean_shift": {"enabled": False},
    "variance_shift": {"enabled": False},
    "trend_shift": {"enabled": False},
    "point_anomaly": {"enabled": False},
    "collective_anomaly": {"enabled": False},
    "contextual_anomaly": {"enabled": False},
}


# ── Generation ────────────────────────────────────────────────────────────────

all_frames: list[pd.DataFrame] = []
series_offset = 0
class_counts: dict[str, int] = {}

for class_name, scenarios in CLASSES.items():
    print(f"\n[{class_name}]")
    class_total = 0

    for i, scenario in enumerate(scenarios):
        base = scenario["base_series"]
        n = scenario["n"]
        feats = {**ALL_FEATURES_OFF, **scenario["features"]}
        seed = RANDOM_SEED + i

        cfg = load_config(
            dataset={
                "base_series": base,
                "num_series": n,
                "length_range": [FIXED_LENGTH, FIXED_LENGTH],
                "random_seed": seed,
                "features": feats,
            }
        )

        df, ctx = generate_dataframe(cfg)

        df["primary_category"] = class_name
        df["sub_category"] = ctx["label"]

        df["series_id"] = df["series_id"] + series_offset
        series_offset = int(df["series_id"].max())

        all_frames.append(df)
        class_total += df["series_id"].nunique()
        print(f"  {ctx['label'][:52]:52s}  base={base:18s}  n={n}")

    class_counts[class_name] = class_total

# ── Merge & shuffle ─────────────────────────────────────────────────────────

print("\nMerging and shuffling...")
combined = pd.concat(all_frames, ignore_index=True)

rng = np.random.default_rng(RANDOM_SEED)
unique_ids = combined["series_id"].unique()
shuffled = rng.permutation(unique_ids)
id_map = {old: new + 1 for new, old in enumerate(shuffled)}
combined["series_id"] = combined["series_id"].map(id_map)
combined = combined.sort_values(["series_id", "time"]).reset_index(drop=True)

# ── Save ────────────────────────────────────────────────────────────────────

out_path = Path(OUTPUT_ROOT) / OUTPUT_NAME
out_path.parent.mkdir(parents=True, exist_ok=True)

for col in combined.select_dtypes(include=["object", "string"]).columns:
    combined[col] = combined[col].astype(str)

combined.to_parquet(out_path, index=False)

# ── Summary ─────────────────────────────────────────────────────────────────

total = combined["series_id"].nunique()
print(f"\n{'─' * 55}")
print(f"  {'Class':<28} {'Series':>8}  {'%':>6}")
print(f"{'─' * 55}")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls:<28} {count:>8,}  {count / total * 100:>5.1f}%")
print(f"{'─' * 55}")
print(f"  {'TOTAL':<28} {total:>8,}  100.0%")
print(f"\nSaved → {out_path}")
print("Class label column : primary_category (= JSON class key)")
print("Fine-grained label : sub_category (BeTiSe ctx label)")
