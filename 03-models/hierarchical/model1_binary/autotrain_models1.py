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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


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

    return {
        "model_path": str(save_path / "model.pkl"),
        "save_path": save_path,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "engine": "pycaret",
        "problem_name": problem_name,
    }


def main():
    parser = argparse.ArgumentParser(description="AutoTrain Model 1 (binary) with AutoGluon or PyCaret")
    parser.add_argument("--engine", type=str, required=True, choices=["autogluon", "pycaret"],
                        help="AutoML engine to use")
    parser.add_argument("--features-path", type=str, default="../../../data/features/selected",
                        help="Path to features directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for train/test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")

    # AutoGluon specific
    parser.add_argument("--time-limit", type=int, default=3600, help="Training time limit (seconds)")
    parser.add_argument("--presets", type=str, default="medium_quality_faster_train",
                        help="AutoGluon presets (e.g., best_quality, medium_quality_faster_train)")

    # PyCaret specific
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds for PyCaret")

    args = parser.parse_args()

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

    # Split
    print("\n[2/4] Train/Test split...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train: {len(X_train_df):,}  Test: {len(X_test_df):,}")

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
    print(f"Test Accuracy : {test_accuracy:.4f} ({100*test_accuracy:.2f}%)")
    if train_accuracy is not None:
        print(f"Train Accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")

    # Compute full report using engine-independent prediction if possible
    try:
        if res["engine"] == "autogluon":
            predictor = res["predictor"]
            y_pred = predictor.predict(X_test_df)
            y_pred_train = predictor.predict(X_train_df)
        else:
            # For PyCaret we already produced y_pred in train function; recompute for report
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
                y_pred_train = preds_df_train["Label"].astype(int).values
            elif "prediction_label" in preds_df_train.columns:
                y_pred_train = preds_df_train["prediction_label"].astype(int).values
            else:
                y_pred_train = model_loaded.predict(X_train_df)
        # Test report
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Stationary", "Non-Stationary"], digits=4))
        print("\nTest Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                   Predicted")
        print(f"                   Stat    Non-Stat")
        print(f"Actual Stat        {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"       Non-Stat    {cm[1,0]:4d}    {cm[1,1]:4d}")

        # Train report
        print("\nTrain Classification Report:")
        print(classification_report(y_train, y_pred_train, target_names=["Stationary", "Non-Stationary"], digits=4))
        print("\nTrain Confusion Matrix:")
        cm_tr = confusion_matrix(y_train, y_pred_train)
        print(f"                   Predicted")
        print(f"                   Stat    Non-Stat")
        print(f"Actual Stat        {cm_tr[0,0]:4d}    {cm_tr[0,1]:4d}")
        print(f"       Non-Stat    {cm_tr[1,0]:4d}    {cm_tr[1,1]:4d}")
    except Exception:
        pass

    # Save common metadata
    metadata = {
        "engine": res.get("engine"),
        "problem_name": res.get("problem_name"),
        "test_accuracy": float(res.get("test_accuracy", np.nan)),
        "train_accuracy": float(res.get("train_accuracy", np.nan)) if res.get("train_accuracy") is not None else None,
        "n_train": int(len(X_train_df)),
        "n_test": int(len(X_test_df)),
        "features_shape": list(X_df.shape),
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


