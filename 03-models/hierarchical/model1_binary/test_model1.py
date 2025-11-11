"""
Test Model 1: Binary Classification
===================================

Quick smoke test for the trained binary classifier. Loads the persisted model,
prepares a small evaluation subset (raw or feature-based), and reports metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Test Model 1: Binary Classification")
parser.add_argument("--n-samples", type=int, default=100,
					help="Number of samples to evaluate (default: 100)")
parser.add_argument("--output-dir", type=str, default=None,
					help="Optional directory to store detailed metrics and predictions")
parser.add_argument("--data-path", type=str, default="../../../data/raw/unified-5k",
					help="Path to raw parquet hierarchy when testing in raw mode")
parser.add_argument("--features-path", type=str, default="../../../data/features/unified-5k/selected",
					help="Path containing features.parquet and labels.parquet for features mode")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else None
if OUTPUT_DIR:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Stationary", "Non-Stationary"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_predict_proba(model, X):
	if hasattr(model, "predict_proba"):
		try:
			return model.predict_proba(X)
		except Exception as exc:  # pragma: no cover -- defensive
			print(f"⚠️  predict_proba unavailable: {exc}")
	return None


def ensure_series_column(df: pd.DataFrame, context: str) -> pd.DataFrame:
	"""Ensure a 'series_id' column exists, renaming legacy 'id' if needed."""
	if "series_id" in df.columns:
		return df
	if "id" in df.columns:
		return df.rename(columns={"id": "series_id"})
	raise ValueError(f"{context} requires a 'series_id' column; columns: {list(df.columns)[:5]}")


def resolve_path(path_str: str) -> Path:
	path = Path(path_str).expanduser()
	if not path.is_absolute():
		path = (Path.cwd() / path).resolve()
	return path


def load_feature_tables(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
	candidates = [
		("standard", features_path / "features.parquet", features_path / "labels.parquet"),
		("binary", features_path / "binary" / "features.parquet", features_path / "binary" / "labels.parquet"),
		("legacy", features_path / "features_binary_mutual_info.parquet", features_path / "labels_binary.parquet"),
	]
	for name, feat_file, lab_file in candidates:
		if feat_file.exists() and lab_file.exists():
			print(f"✓ Using {name} features: {feat_file}")
			print(f"✓ Using {name} labels: {lab_file}")
			X_df = pd.read_parquet(feat_file)
			labels_df = pd.read_parquet(lab_file)
			return X_df, labels_df, name
	raise FileNotFoundError(
		"Expected features.parquet + labels.parquet (standard), binary/ directory, or legacy mutual-info files"
	)


def prepare_fixed_length(series: np.ndarray, target_length: int) -> np.ndarray:
	if len(series) > target_length:
		start = (len(series) - target_length) // 2
		return series[start:start + target_length]
	if len(series) < target_length:
		pad_width = target_length - len(series)
		return np.pad(series, (0, pad_width), mode="constant")
	return series


# ---------------------------------------------------------------------------
# Load model metadata
# ---------------------------------------------------------------------------
print("=" * 80)
print("MODEL 1 TEST - Binary Classification")
print("=" * 80)

model_path = Path("saved_models/model1_binary_classifier.pkl")
metadata_path = Path("saved_models/model1_metadata.pkl")

if not model_path.exists():
	print("❌ Model not found. Train before testing: python train_model1.py")
	raise SystemExit(1)

print("\n[1/3] Loading model...")
with model_path.open("rb") as fh:
	model = pickle.load(fh)
with metadata_path.open("rb") as fh:
	metadata = pickle.load(fh)

mode = metadata.get("mode", "raw")
print(f"✓ Loaded model: {metadata.get('model_name', 'Unknown')}")
print(f"  Mode: {mode}")
print(f"  Training accuracy: {100 * metadata.get('accuracy', 0):.2f}%")
if mode == "raw":
	print(f"  Feature shape: {metadata.get('feature_shape', 'N/A')}")
else:
	print(f"  Number of features: {metadata.get('n_features', 'N/A')}")

# ---------------------------------------------------------------------------
# Prepare evaluation data
# ---------------------------------------------------------------------------
print("\n[2/3] Loading test data...")

if mode == "raw":
	data_path = resolve_path(args.data_path)
	if not data_path.exists():
		print(f"❌ Error: Test data not found: {data_path}")
		raise SystemExit(1)

	parquet_files = list(data_path.rglob("*.parquet"))
	if not parquet_files:
		print(f"❌ Error: No parquet files under {data_path}")
		raise SystemExit(1)

	print(f"Found {len(parquet_files)} parquet files")

	dfs = []
	for fp in parquet_files[:5]:  # keep load light for smoke testing
		part_df = pd.read_parquet(fp)
		dfs.append(part_df)
	raw_df = pd.concat(dfs, ignore_index=True)
	raw_df = ensure_series_column(raw_df, "Raw evaluation DataFrame")

	fixed_length = metadata.get("fixed_length", 1500)
	test_series = []
	test_labels = []
	sample_ids = []

	for idx, series_id in enumerate(raw_df["series_id"].unique()):
		if idx >= args.n_samples:
			break
		series_df = raw_df[raw_df["series_id"] == series_id].sort_values("time")
		if "data" in series_df.columns:
			values = series_df["data"].to_numpy()
		elif "value" in series_df.columns:
			values = series_df["value"].to_numpy()
		else:
			values = series_df.iloc[:, 2].to_numpy()
		label = 0 if series_df["is_stationary"].iloc[0] else 1
		test_series.append(prepare_fixed_length(values, fixed_length))
		test_labels.append(label)
		sample_ids.append(series_id)

	X_test = np.asarray(test_series).reshape(-1, 1, fixed_length)
	y_test = np.asarray(test_labels)
	sample_ids = np.asarray(sample_ids)

else:
	features_path = resolve_path(args.features_path)
	try:
		X_df, labels_df, source = load_feature_tables(features_path)
	except FileNotFoundError as exc:
		print(f"❌ Error: {exc}")
		raise SystemExit(1)

	X_df = ensure_series_column(X_df, f"{source} features table")
	labels_df = ensure_series_column(labels_df, f"{source} labels table")

	X_df = X_df.set_index("series_id").head(args.n_samples)
	labels_df = labels_df.set_index("series_id").reindex(X_df.index)

	if labels_df.isnull().any().any():
		missing_ids = labels_df.index[labels_df.isnull().any(axis=1)].tolist()
		print(f"❌ Error: Missing labels for series IDs: {missing_ids[:5]}")
		raise SystemExit(1)

	if "is_stationary" in labels_df.columns:
		y_test = (labels_df["is_stationary"] == False).astype(int).to_numpy()
	elif "label" in labels_df.columns:
		y_test = labels_df["label"].astype(int).to_numpy()
	else:
		print("❌ Error: Labels must include 'is_stationary' or 'label' column")
		raise SystemExit(1)

	print(f"✓ Loaded features: {X_df.shape}")

	scaler = metadata.get("scaler")
	if scaler is not None:
		X_test = scaler.transform(X_df.to_numpy())
	else:
		X_test = X_df.to_numpy()
	sample_ids = X_df.index.to_numpy()

print(f"✓ Prepared {len(X_test)} test samples")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
print("\n[3/3] Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_proba = safe_predict_proba(model, X_test)

print(f"\n✓ Test Accuracy: {100 * accuracy:.2f}%")
missing_classes = [idx for idx in range(len(CLASS_NAMES)) if idx not in np.unique(y_test)]
if missing_classes:
	missing_names = ", ".join(CLASS_NAMES[idx] for idx in missing_classes)
	print(f"\n⚠️  Test subset lacks samples for: {missing_names}")

print("\nClassification Report:")
report_text = classification_report(
	y_test,
	y_pred,
	labels=list(range(len(CLASS_NAMES))),
	target_names=CLASS_NAMES,
	digits=4,
	zero_division=0,
)
print(report_text)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
print("                   Predicted")
print("                   Stat    Non-Stat")
print(f"Actual Stat        {cm[0, 0]:4d}    {cm[0, 1]:4d}")
print(f"       Non-Stat    {cm[1, 0]:4d}    {cm[1, 1]:4d}")

report_dict = classification_report(
	y_test,
	y_pred,
	labels=list(range(len(CLASS_NAMES))),
	target_names=CLASS_NAMES,
	output_dict=True,
	zero_division=0,
)

misclassified_mask = y_pred != y_test
misclassified_count = int(np.sum(misclassified_mask))
if misclassified_count:
	print(f"\n⚠️  Misclassified samples: {misclassified_count}")
	preview_ids = sample_ids[misclassified_mask][:5]
	print(f"    Examples: {preview_ids}")
else:
	print("\nNo misclassifications detected in this subset.")

predictions_df = pd.DataFrame({
	"sample_id": sample_ids,
	"true_label": y_test,
	"predicted_label": y_pred,
})

if y_proba is not None:
	proba_array = np.asarray(y_proba)
	for idx, cls_name in enumerate(CLASS_NAMES):
		col_name = f"prob_{cls_name.lower().replace(' ', '_')}"
		predictions_df[col_name] = proba_array[:, idx]

misclassified_df = predictions_df[predictions_df["true_label"] != predictions_df["predicted_label"]]

results_payload = {
	"timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
	"mode": mode,
	"model_name": metadata.get("model_name"),
	"n_samples": int(len(X_test)),
	"test_accuracy": float(accuracy),
	"classification_report": report_dict,
	"confusion_matrix": cm.tolist(),
	"misclassified_count": misclassified_count,
	"probability_available": y_proba is not None,
}

if OUTPUT_DIR:
	metrics_path = OUTPUT_DIR / f"model1_test_metrics_{mode}.json"
	predictions_path = OUTPUT_DIR / f"model1_test_predictions_{mode}.csv"
	predictions_df.to_csv(predictions_path, index=False)
	results_payload["artifacts"] = {
		"metrics_json": str(metrics_path),
		"predictions_csv": str(predictions_path),
	}
	if not misclassified_df.empty:
		misclassified_path = OUTPUT_DIR / f"model1_test_misclassified_{mode}.csv"
		misclassified_df.to_csv(misclassified_path, index=False)
		results_payload["artifacts"]["misclassified_csv"] = str(misclassified_path)
	with metrics_path.open("w", encoding="utf-8") as fh:
		json.dump(results_payload, fh, indent=2)
else:
	results_payload["artifacts"] = {}

print("=" * 80)
print("✅ TEST COMPLETE!")
print("=" * 80)
print(f"\nTested {len(X_test)} samples")
print(f"Mode: {mode.upper()}")
print(f"Model: {metadata.get('model_name', 'Unknown')}")
print(f"Accuracy: {100 * accuracy:.2f}%")
print("=" * 80)

