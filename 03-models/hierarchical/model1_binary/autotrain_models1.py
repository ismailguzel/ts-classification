"""
AutoTrain Model 1 (Binary): Stationary vs Non-Stationary
=========================================================

Trains an automatic tabular learner using extracted features via:
- engine=autogluon  → AutoGluon Tabular
- engine=pycaret    → PyCaret Classification

Input features are loaded from --features-path with robust fallbacks.
Outputs are saved under saved_models/ with engine-specific artifacts and
common metadata for reproducibility.

Usage examples:
    python autotrain_models1.py --engine autogluon \
        --features-path ../../../data/features/selected \
        --time-limit 3600 --presets medium_quality_faster_train

    python autotrain_models1.py --engine pycaret \
        --features-path ../../../data/features/selected \
        --folds 5
"""

import argparse
import warnings
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

CLASS_NAMES = ['Stationary', 'Non-Stationary']


def safe_predict_proba(model, X):
    """Return predict_proba output when available; otherwise None."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def align_probability_matrix(proba, class_labels):
    """Align probability outputs to expected class label order."""
    if proba is None:
        return None
    if isinstance(proba, pd.DataFrame):
        aligned_cols = []
        for label in class_labels:
            if label in proba.columns:
                aligned_cols.append(label)
            elif str(label) in proba.columns:
                aligned_cols.append(str(label))
        if aligned_cols:
            proba = proba[aligned_cols]
        return proba.to_numpy()
    return np.asarray(proba)


def load_features_and_labels(features_path: Path):
    """
    Load features and labels for binary classification with robust fallbacks.

    Priority order:
      1) features.parquet + labels.parquet
      2) binary/features.parquet + binary/labels.parquet
      3) features_binary_mutual_info.parquet + labels_binary.parquet

    Label mapping:
      - If 'is_stationary' exists → y = (~is_stationary).astype(int)
      - Else if 'label' exists → y = label
    """
    # 1) Standard at root
    feat_file = features_path / "features.parquet"
    lab_file = features_path / "labels.parquet"

    # 2) Standard in binary/
    if not (feat_file.exists() and lab_file.exists()):
        feat_file = features_path / "binary" / "features.parquet"
        lab_file = features_path / "binary" / "labels.parquet"

    # 3) Legacy names
    if not (feat_file.exists() and lab_file.exists()):
        feat_file = features_path / "features_binary_mutual_info.parquet"
        lab_file = features_path / "labels_binary.parquet"

    if not (feat_file.exists() and lab_file.exists()):
        raise FileNotFoundError(
            f"Could not find features/labels under {features_path}. Tried standard and legacy names."
        )

    X_df = pd.read_parquet(feat_file)
    y_df = pd.read_parquet(lab_file)

    if "id" in X_df.columns:
        X_df = X_df.set_index("id")

    if "is_stationary" in y_df.columns:
        y = (y_df["is_stationary"] == False).astype(int).values
    elif "label" in y_df.columns:
        y = y_df["label"].astype(int).values
    else:
        raise ValueError("Label column not found in labels file (expected 'is_stationary' or 'label').")

    return X_df, y


def train_autogluon(X_train_df: pd.DataFrame, y_train: np.ndarray, X_test_df: pd.DataFrame, y_test: np.ndarray,
                    save_dir: Path, time_limit: int, presets: str, problem_name: str):
    from autogluon.tabular import TabularPredictor

    label_col = "_label"
    train_df = X_train_df.copy()
    train_df[label_col] = y_train
    test_df = X_test_df.copy()
    test_df[label_col] = y_test

    save_path = save_dir / "model1_binary_autogluon"
    save_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(label=label_col, problem_type="binary", path=str(save_path))
    fit_kwargs = {
        "train_data": train_df,
        "time_limit": time_limit if time_limit is not None else None,
    }
    if presets:
        fit_kwargs["presets"] = presets

    start_time = time.time()
    predictor.fit(**fit_kwargs)
    train_time = time.time() - start_time

    # Evaluation
    y_pred_test = predictor.predict(X_test_df)
    test_acc = accuracy_score(y_test, y_pred_test)
    y_pred_train = predictor.predict(X_train_df)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Save leaderboards for train and test
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

    return {
        "predictor": predictor,
        "save_path": save_path,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "train_time": train_time,
        "leaderboard_test_path": str(lb_test_path) if lb_test_path is not None else None,
        "leaderboard_train_path": str(lb_train_path) if lb_train_path is not None else None,
        "engine": "autogluon",
        "problem_name": problem_name,
    }


def train_pycaret(X_train_df: pd.DataFrame, y_train: np.ndarray, X_test_df: pd.DataFrame, y_test: np.ndarray,
                  save_dir: Path, folds: int, problem_name: str):
    from pycaret.classification import setup, compare_models, finalize_model, predict_model, save_model, pull

    label_col = "_label"
    train_df = X_train_df.copy()
    train_df[label_col] = y_train

    start_time = time.time()
    # PyCaret setup (PyCaret 3.x doesn't accept 'silent')
    setup(data=train_df, target=label_col, fold=folds, session_id=42, verbose=False)
    best_model = compare_models()  # picks best based on default metric
    # Save leaderboard from compare_models
    try:
        leaderboard_df = pull()
    except Exception:
        leaderboard_df = None
    best_final = finalize_model(best_model)

    # Evaluate on holdout (PyCaret's internal) and our test set
    _ = predict_model(best_final)  # ensures metrics are computed
    try:
        holdout_results = pull()
    except Exception:
        holdout_results = None

    preds = predict_model(best_final, data=X_test_df)
    # predict_model returns a df with prediction column depending on pycaret version
    if "Label" in preds.columns:
        y_pred = preds["Label"].astype(int).values
    elif "prediction_label" in preds.columns:
        y_pred = preds["prediction_label"].astype(int).values
    else:
        # Fallback to model's native predict
        y_pred = best_final.predict(X_test_df)

    test_acc = accuracy_score(y_test, y_pred)

    # Train performance
    preds_train = predict_model(best_final, data=X_train_df)
    if "Label" in preds_train.columns:
        y_pred_train = preds_train["Label"].astype(int).values
    elif "prediction_label" in preds_train.columns:
        y_pred_train = preds_train["prediction_label"].astype(int).values
    else:
        y_pred_train = best_final.predict(X_train_df)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Save
    save_path = save_dir / "model1_binary_pycaret"
    save_path.mkdir(parents=True, exist_ok=True)
    save_model(best_final, str(save_path / "model"))  # creates model.pkl & transformer if any

    # Save leaderboards and holdout results
    if leaderboard_df is not None:
        try:
            leaderboard_df.to_csv(save_path / "leaderboard.csv", index=False)
        except Exception:
            pass
    if holdout_results is not None:
        try:
            holdout_results.to_csv(save_path / "holdout_results.csv", index=False)
        except Exception:
            pass

    train_time = time.time() - start_time

    return {
        "model_path": str(save_path / "model.pkl"),
        "save_path": save_path,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "train_time": train_time,
        "engine": "pycaret",
        "problem_name": problem_name,
    }


def main():
    parser = argparse.ArgumentParser(description="AutoTrain Model 1 (binary) with AutoGluon or PyCaret")
    parser.add_argument("--engine", type=str, required=True, choices=["autogluon", "pycaret"],
                        help="AutoML engine to use")
    parser.add_argument("--features-path", type=str, default="../../../data/features/unified-5k/selected",
                        help="Path to features directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for train/test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional directory for structured metrics and predictions")

    # AutoGluon specific
    parser.add_argument("--time-limit", type=int, default=3600, help="Training time limit (seconds)")
    parser.add_argument("--presets", type=str, default="medium_quality_faster_train",
                        help="AutoGluon presets (e.g., best_quality, medium_quality_faster_train)")

    # PyCaret specific
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds for PyCaret")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL 1 AUTOTRAIN - Binary Classification")
    print("=" * 80)
    print(f"Engine       : {args.engine}")
    print(f"Features path: {args.features_path}")
    print(f"Test size    : {args.test_size}")
    print("=" * 80)

    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"❌ Error: Features path not found: {features_path}")
        return

    # Load data
    print("\n[1/4] Loading features and labels...")
    X_df, y = load_features_and_labels(features_path)
    print(f"✓ Features: {X_df.shape}")
    print(f"✓ Samples : {len(y)}")

    class_labels = sorted(np.unique(y))
    class_names_ordered = [CLASS_NAMES[int(lbl)] for lbl in class_labels]

    # Split
    print("\n[2/4] Train/Test split...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train: {len(X_train_df):,}  Test: {len(X_test_df):,}")

    train_ids = X_train_df.index.to_numpy()
    test_ids = X_test_df.index.to_numpy()

    # Train
    print("\n[3/4] Training...")
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    problem_name = "model1_binary"

    if args.engine == "autogluon":
        try:
            res = train_autogluon(
                X_train_df=X_train_df,
                y_train=y_train,
                X_test_df=X_test_df,
                y_test=y_test,
                save_dir=save_dir,
                time_limit=args.time_limit,
                presets=args.presets,
                problem_name=problem_name,
            )
        except ImportError as e:
            print(f"❌ AutoGluon not installed: {e}")
            return
    else:
        try:
            res = train_pycaret(
                X_train_df=X_train_df,
                y_train=y_train,
                X_test_df=X_test_df,
                y_test=y_test,
                save_dir=save_dir,
                folds=args.folds,
                problem_name=problem_name,
            )
        except ImportError as e:
            print(f"❌ PyCaret not installed: {e}")
            return

    # Evaluate & report
    print("\n[4/4] Evaluation")
    print("=" * 80)
    test_accuracy = res.get("test_accuracy")
    train_accuracy = res.get("train_accuracy")
    train_time_value = res.get("train_time")
    print(f"Test Accuracy : {test_accuracy:.4f} ({100*test_accuracy:.2f}%)")
    if train_accuracy is not None:
        print(f"Train Accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")
    if train_time_value is not None:
        print(f"Train Time    : {train_time_value:.2f}s")

    y_pred = None
    y_train_pred = None
    y_proba = None
    y_train_proba = None
    cm = None
    cm_tr = None
    test_report_dict = {}
    train_report_dict = {}
    predictions_df = pd.DataFrame()
    misclassified_df = pd.DataFrame()
    misclassified_count = 0
    probability_available = False

    try:
        if res["engine"] == "autogluon":
            predictor = res["predictor"]
            y_pred = predictor.predict(X_test_df)
            y_train_pred = predictor.predict(X_train_df)
            y_proba = align_probability_matrix(predictor.predict_proba(X_test_df), class_labels)
            y_train_proba = align_probability_matrix(predictor.predict_proba(X_train_df), class_labels)
        else:
            from pycaret.classification import load_model, predict_model

            model_dir = Path(res["save_path"]) / "model"
            model_loaded = load_model(str(model_dir))

            preds_df = predict_model(model_loaded, data=X_test_df)
            if "Label" in preds_df.columns:
                y_pred = preds_df["Label"].astype(int).values
            elif "prediction_label" in preds_df.columns:
                y_pred = preds_df["prediction_label"].astype(int).values
            else:
                y_pred = model_loaded.predict(X_test_df)

            preds_df_train = predict_model(model_loaded, data=X_train_df)
            if "Label" in preds_df_train.columns:
                y_train_pred = preds_df_train["Label"].astype(int).values
            elif "prediction_label" in preds_df_train.columns:
                y_train_pred = preds_df_train["prediction_label"].astype(int).values
            else:
                y_train_pred = model_loaded.predict(X_train_df)

            y_proba = align_probability_matrix(safe_predict_proba(model_loaded, X_test_df), class_labels)
            y_train_proba = align_probability_matrix(safe_predict_proba(model_loaded, X_train_df), class_labels)

        if hasattr(y_pred, "to_numpy"):
            y_pred = y_pred.to_numpy()
        if hasattr(y_train_pred, "to_numpy"):
            y_train_pred = y_train_pred.to_numpy()

        if y_pred is not None:
            y_pred = np.asarray(y_pred)
            y_train_pred = np.asarray(y_train_pred) if y_train_pred is not None else None

            report_text = classification_report(
                y_test,
                y_pred,
                target_names=class_names_ordered,
                digits=4
            )
            print("\nTest Classification Report:")
            print(report_text)

            cm = confusion_matrix(y_test, y_pred)
            print("\nTest Confusion Matrix:")
            print(f"                   Predicted")
            print(f"                   Stat    Non-Stat")
            print(f"Actual Stat        {cm[0,0]:4d}    {cm[0,1]:4d}")
            print(f"       Non-Stat    {cm[1,0]:4d}    {cm[1,1]:4d}")

            test_report_dict = classification_report(
                y_test,
                y_pred,
                target_names=class_names_ordered,
                output_dict=True,
                zero_division=0
            )

            if y_train_pred is not None:
                print("\nTrain Classification Report:")
                print(classification_report(
                    y_train,
                    y_train_pred,
                    target_names=class_names_ordered,
                    digits=4
                ))

                cm_tr = confusion_matrix(y_train, y_train_pred)
                print("\nTrain Confusion Matrix:")
                print(f"                   Predicted")
                print(f"                   Stat    Non-Stat")
                print(f"Actual Stat        {cm_tr[0,0]:4d}    {cm_tr[0,1]:4d}")
                print(f"       Non-Stat    {cm_tr[1,0]:4d}    {cm_tr[1,1]:4d}")

                train_report_dict = classification_report(
                    y_train,
                    y_train_pred,
                    target_names=class_names_ordered,
                    output_dict=True,
                    zero_division=0
                )

            predictions_df = pd.DataFrame({
                "sample_id": test_ids,
                "true_label": y_test,
                "predicted_label": y_pred,
            })

            if y_proba is not None:
                prob_array = np.asarray(y_proba)
                if prob_array.ndim == 1:
                    prob_array = prob_array.reshape(-1, 1)
                probability_available = prob_array.ndim == 2 and prob_array.shape[1] == len(class_names_ordered)
                if probability_available:
                    for idx, cls_name in enumerate(class_names_ordered):
                        col_name = f"prob_{cls_name.lower().replace(' ', '_').replace('-', '_')}"
                        predictions_df[col_name] = prob_array[:, idx]
            else:
                probability_available = False

            misclassified_mask = predictions_df["true_label"] != predictions_df["predicted_label"]
            misclassified_count = int(misclassified_mask.sum())
            if misclassified_count > 0:
                preview_ids = predictions_df.loc[misclassified_mask, "sample_id"].head(5).tolist()
                print(f"\n⚠️  Misclassified samples: {misclassified_count} (examples: {preview_ids})")
            else:
                print("\nNo misclassifications detected on the hold-out set.")

            misclassified_df = predictions_df.loc[misclassified_mask].copy()
    except Exception as exc:
        print(f"⚠️  Detailed evaluation skipped due to error: {exc}")

    metrics_payload = {
        "engine": res.get("engine"),
        "problem_name": problem_name,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "train_time_sec": float(train_time_value) if isinstance(train_time_value, (int, float)) else None,
        "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
        "train_accuracy": float(train_accuracy) if train_accuracy is not None else None,
        "n_train": int(len(X_train_df)),
        "n_test": int(len(X_test_df)),
        "train_sample_ids": train_ids.tolist(),
        "test_sample_ids": test_ids.tolist(),
        "class_labels": [int(lbl) for lbl in class_labels],
        "class_names": class_names_ordered,
        "test_confusion_matrix": cm.tolist() if cm is not None else None,
        "train_confusion_matrix": cm_tr.tolist() if cm_tr is not None else None,
        "test_classification_report": test_report_dict,
        "train_classification_report": train_report_dict,
        "misclassified_count": misclassified_count,
        "probability_available": probability_available,
        "artifacts": {},
    }

    if output_dir:
        artifacts = {}
        metrics_path = output_dir / f"{problem_name}_{args.engine}_metrics.json"

        if not predictions_df.empty:
            predictions_path = output_dir / f"{problem_name}_{args.engine}_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            artifacts["predictions_csv"] = str(predictions_path)

            if not misclassified_df.empty:
                misclassified_path = output_dir / f"{problem_name}_{args.engine}_misclassified.csv"
                misclassified_df.to_csv(misclassified_path, index=False)
                artifacts["misclassified_csv"] = str(misclassified_path)

        metrics_payload["artifacts"] = artifacts

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        print(f"Metrics JSON : {metrics_path}")
        if "predictions_csv" in artifacts:
            print(f"Predictions  : {artifacts['predictions_csv']}")
        if "misclassified_csv" in artifacts:
            print(f"Misclassified: {artifacts['misclassified_csv']}")
    else:
        metrics_payload["artifacts"] = {}

    # Save common metadata
    metadata = {
        "engine": res.get("engine"),
        "problem_name": res.get("problem_name"),
        "test_accuracy": float(res.get("test_accuracy", np.nan)),
        "train_accuracy": float(res.get("train_accuracy", np.nan)) if res.get("train_accuracy") is not None else None,
        "n_train": int(len(X_train_df)),
        "n_test": int(len(X_test_df)),
        "features_shape": list(X_df.shape),
        "class_names": class_names_ordered,
        "train_time": float(train_time_value) if isinstance(train_time_value, (int, float)) else None,
        "presets": None,
        "time_limit": None,
        "folds": None,
    }
    if args.engine == "autogluon":
        metadata["presets"] = args.presets
        metadata["time_limit"] = args.time_limit
        metadata["artifact_path"] = str(res.get("save_path"))
        metadata["leaderboard_test_path"] = res.get("leaderboard_test_path")
        metadata["leaderboard_train_path"] = res.get("leaderboard_train_path")
    else:
        metadata["folds"] = args.folds
        metadata["artifact_path"] = str(res.get("save_path"))

    if output_dir:
        metadata["metrics_output_dir"] = str(output_dir)

    meta_path = save_dir / "model1_autotrain_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 80)
    print("✅ AUTOTRAIN COMPLETE")
    print("=" * 80)
    print(f"Engine     : {metadata['engine']}")
    print(f"Test Acc   : {100*metadata['test_accuracy']:.2f}%")
    if metadata.get("train_accuracy") is not None:
        print(f"Train Acc  : {100*metadata['train_accuracy']:.2f}%")
    print(f"Artifacts  : {metadata['artifact_path']}")
    print(f"Metadata   : {meta_path}")


if __name__ == "__main__":
    main()


