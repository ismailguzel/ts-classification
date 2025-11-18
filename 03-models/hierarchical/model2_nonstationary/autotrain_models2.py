"""
AutoTrain Model 2 (5-Class): Primary Category Classification
=============================================================

Refactored training entry-point that relies on the shared utils module for:
- feature/label loading (`load_features_and_labels` with target='primary')
- leakage-safe splits (`remove_series_id_leakage`, `split_train_test`)
- evaluation + reporting (`ModelEvaluator`)

Supports AutoGluon for automated 5-class classification.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from utils import (  # noqa: E402
    ModelEvaluator,
    load_features_and_labels,
    remove_series_id_leakage,
    split_train_test,
    PRIMARY_CATEGORY_MAPPING,
    PRIMARY_CLASS_NAMES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)

warnings.filterwarnings("ignore")


@dataclass
class TrainingResult:
    """Container that standardizes the objects returned by each engine."""

    engine: str
    problem_name: str
    metrics: Dict[str, Any]
    save_path: Path
    train_time: float
    model: Any
    artifacts: Dict[str, Optional[str]]


class AutoGluonModelAdapter:
    """Ensures AutoGluon predictors expose numpy-friendly outputs."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, X):
        preds = self.predictor.predict(X)
        return preds.to_numpy() if hasattr(preds, "to_numpy") else preds

    def predict_proba(self, X):
        proba = self.predictor.predict_proba(X, as_multiclass=True)
        if hasattr(proba, "values"):
            proba = proba.values
        return proba


def _print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print a brief summary of model evaluation metrics."""
    train_metrics = metrics.get("metrics", {}).get("train", {})
    test_metrics = metrics.get("metrics", {}).get("test", {})

    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    if train_metrics:
        print(f"Train Accuracy : {train_metrics.get('accuracy', float('nan')):.4f}")
        print(f"Train F1       : {train_metrics.get('f1', float('nan')):.4f}")
    if test_metrics:
        print(f"Test Accuracy  : {test_metrics.get('accuracy', float('nan')):.4f}")
        print(f"Test F1        : {test_metrics.get('f1', float('nan')):.4f}")
    roc_metrics = metrics.get("roc_metrics", {}).get("test", {})
    if roc_metrics:
        print(f"Test ROC-AUC   : {roc_metrics.get('roc_auc', float('nan')):.4f}")
    print("=" * 80 + "\n")


def train_autogluon(
    *,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    save_dir: Path,
    time_limit: Optional[int],
    presets: Optional[str],
    problem_name: str,
    train_ids: Sequence[Any],
    test_ids: Sequence[Any],
) -> TrainingResult:
    """Train an AutoGluon TabularPredictor for 5-class classification."""

    from autogluon.tabular import TabularPredictor

    label_col = "_label"
    X_train_clean = remove_series_id_leakage(X_train_df)
    X_test_clean = remove_series_id_leakage(X_test_df)

    train_df = X_train_clean.copy()
    train_df[label_col] = y_train
    test_df = X_test_clean.copy()
    test_df[label_col] = y_test

    # Use save_dir as the unified output directory
    save_path = save_dir / "model2_nonstationary_autogluon"
    save_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(label=label_col, problem_type="multiclass", path=str(save_path))
    fit_kwargs: Dict[str, Any] = {"train_data": train_df}
    if time_limit:
        fit_kwargs["time_limit"] = time_limit
    if presets:
        fit_kwargs["presets"] = presets

    start_time = time.time()
    predictor.fit(**fit_kwargs)
    train_time = time.time() - start_time
    print(f"\n✓ AutoGluon training completed in {train_time:.1f}s")

    # Use save_path for all outputs (model, metrics, predictions)
    evaluator = (
        ModelEvaluator(
            model_name="AutoGluon_5Class",
            model=AutoGluonModelAdapter(predictor),
            class_names=PRIMARY_CLASS_NAMES,
            output_dir=save_path,  # Save predictions to same directory
        )
        .set_train_time(train_time)
        .set_feature_names(list(X_train_clean.columns))
    )
    metrics = evaluator.evaluate(
        X_train_clean,
        y_train,
        X_test_clean,
        y_test,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    
    # Save comprehensive metrics (JSON format from ModelEvaluator)
    metrics_dir = save_path / "metrics"
    evaluator.save_results(metrics_dir)
    
    # Extract feature importance from AutoGluon
    print("\n[Extracting Feature Importance from AutoGluon...]")
    try:
        # Get feature importance (returns DataFrame with feature names as index)
        feature_importance = predictor.feature_importance(test_df)
        
        if feature_importance is not None and not feature_importance.empty:
            # AutoGluon returns DataFrame with columns ['importance', 'stddev', 'p_value', 'n', 'p99_high', 'p99_low']
            # We only need 'importance' column
            
            # Sort by importance column (descending) and get top 50
            if 'importance' in feature_importance.columns:
                feature_importance_sorted = feature_importance.sort_values(
                    by='importance', ascending=False
                ).head(50)
                
                # Keep only feature name and importance
                fi_df = feature_importance_sorted[['importance']].copy()
                fi_df.index.name = 'feature_name'
                
                # Save to CSV
                fi_path = save_path / "feature_importance_AutoGluon.csv"
                fi_df.to_csv(fi_path)
                print(f"✓ Feature importance saved to: {fi_path}")
                print(f"  Top 5 features:")
                for idx, (feat, row) in enumerate(feature_importance_sorted.head(5).iterrows(), 1):
                    feat_display = feat.replace('data__', '').replace('__', ' ')[:55]
                    print(f"    {idx}. {feat_display}: {row['importance']:.4f}")
            else:
                print("  ⚠ 'importance' column not found in feature_importance DataFrame")
        else:
            print("  ⚠ Feature importance not available from AutoGluon")
    except Exception as e:
        print(f"  ⚠ Could not extract feature importance: {e}")

    lb_test_path = None
    lb_train_path = None
    try:
        lb_test = predictor.leaderboard(test_df, silent=True)
        lb_test_path = save_path / "leaderboard_test.csv"
        lb_test.to_csv(lb_test_path, index=False)
    except Exception:
        pass
    try:
        lb_train = predictor.leaderboard(train_df, silent=True)
        lb_train_path = save_path / "leaderboard_train.csv"
        lb_train.to_csv(lb_train_path, index=False)
    except Exception:
        pass

    artifacts = {
        "leaderboard_test_path": str(lb_test_path) if lb_test_path else None,
        "leaderboard_train_path": str(lb_train_path) if lb_train_path else None,
        "metrics_dir": str(metrics_dir),
    }

    return TrainingResult(
        engine="autogluon",
        problem_name=problem_name,
        metrics=metrics,
        save_path=save_path,
        train_time=train_time,
        model=predictor,
        artifacts=artifacts,
    )


def main():
    parser = argparse.ArgumentParser(description="AutoTrain Model 2 (5-class) with AutoGluon")
    parser.add_argument("--features-path", type=str, default="../../../data/features/unified-5k/selected",
                        help="Path to features directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for train/test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--save-dir", type=str, default="saved_models",
                        help="Directory to save model, metrics, predictions, and leaderboards")

    # AutoGluon specific
    parser.add_argument("--time-limit", type=int, default=3600, help="Training time limit (seconds)")
    parser.add_argument("--presets", type=str, default="medium_quality_faster_train",
                        help="AutoGluon presets (e.g., best_quality, medium_quality_faster_train)")

    args = parser.parse_args()

    print("=" * 80)
    print("MODEL 2 AUTOTRAIN - 5-Class Non-Stationary Classification (AutoGluon)")
    print("=" * 80)
    print(f"Features path: {args.features_path}")
    print(f"Test size    : {args.test_size}")
    print("=" * 80)

    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f" Error: Features path not found: {features_path}")
        return

    # Load data
    print("\n[1/3] Loading features and labels...")
    X_df, y, _ = load_features_and_labels(features_path, target="primary", verbose=True)
    print(f"✓ Features shape: {X_df.shape}")
    print(f"✓ Samples       : {len(y)}")

    class_labels = sorted(np.unique(y))

    # Split
    print("\n[2/3] Train/Test split...")
    (
        X_train_df,
        X_test_df,
        y_train,
        y_test,
        train_ids,
        test_ids,
    ) = split_train_test(
        X_df,
        y,
        sample_ids=X_df.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )
    print(f"Train: {len(X_train_df):,}  Test: {len(X_test_df):,}")

    # Train
    print("\n[3/3] Training with AutoGluon...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    problem_name = "model2_primary"

    try:
        result = train_autogluon(
            X_train_df=X_train_df,
            y_train=y_train,
            X_test_df=X_test_df,
            y_test=y_test,
            save_dir=save_dir,
            time_limit=args.time_limit,
            presets=args.presets,
            problem_name=problem_name,
            train_ids=train_ids,
            test_ids=test_ids,
        )
    except ImportError as exc:
        print(f" Required package missing: {exc}")
        return

    _print_metrics_summary(result.metrics)

    # Create metadata for reference
    metadata = {
        "engine": result.engine,
        "problem_name": result.problem_name,
        "n_train": len(X_train_df),
        "n_test": len(X_test_df),
        "features_shape": list(X_df.shape),
        "class_names": PRIMARY_CLASS_NAMES,
        "train_time": float(result.train_time),
        "artifact_path": str(result.save_path),
        "metrics_dir": result.artifacts.get("metrics_dir"),
        "leaderboard_test_path": result.artifacts.get("leaderboard_test_path"),
        "leaderboard_train_path": result.artifacts.get("leaderboard_train_path"),
        "presets": args.presets,
        "time_limit": args.time_limit,
    }

    # Save metadata inside AutoGluon directory to avoid mixing with raw/features mode
    meta_path = result.save_path / "model2_autotrain_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 80)
    print(" AUTOTRAIN COMPLETE")
    print("=" * 80)
    print(f"Engine     : {metadata['engine']}")
    if result.metrics.get('metrics', {}).get('train'):
        print(
            f"Train Acc  : {result.metrics['metrics']['train'].get('accuracy', float('nan')):.4f}"
        )
    if result.metrics.get('metrics', {}).get('test'):
        print(
            f"Test Acc   : {result.metrics['metrics']['test'].get('accuracy', float('nan')):.4f}"
        )
    print(f"Artifacts  : {metadata['artifact_path']}")
    print(f"Metadata   : {meta_path}")
    print("=" * 80)
    
    # Print detailed output structure
    print("\n Output Structure:")
    print(f"   {save_dir}/")
    print(f"   └── {problem_name}/                       # AutoGluon model artifacts")
    print(f"       ├── models/                           # Trained model files")
    print(f"       ├── predictions_AutoGluon_5Class.csv  # Test predictions with probabilities")
    print(f"       ├── misclassified_AutoGluon_5Class.csv# Misclassified samples")
    print(f"       ├── leaderboard_test.csv              # Test performance leaderboard")
    print(f"       ├── leaderboard_train.csv             # Train performance leaderboard")
    print(f"       ├── model2_autotrain_metadata.pkl     # Training metadata")
    print(f"       └── metrics/")
    print(f"           └── AutoGluon_5Class_metrics.json # Detailed metrics")


if __name__ == "__main__":
    main()


