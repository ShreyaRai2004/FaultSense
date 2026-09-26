from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.model import train_classifier, train_baseline, train_anomaly_detector

DATA = Path("data/ai4i2020.csv")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_failure": float(precision_score(y, pred, zero_division=0)),
        "recall_failure": float(recall_score(y, pred, zero_division=0)),
        "f1_failure": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError("data/ai4i2020.csv not found. Run: python download_data.py")
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    from src.preprocess import clean_columns
    df = clean_columns(df)
    print("Detected dataset columns:", ", ".join(df.columns))
    # The dataset is ordered by UDI, but we use a stratified holdout to make the
    # benchmark reproducible. We never use failure-mode columns as input features.
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["Machine failure"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.20, stratify=train_df["Machine failure"], random_state=42
    )

    from src.model import make_training_frame
    X_train, y_train = make_training_frame(train_df)
    X_val, y_val = make_training_frame(val_df)
    X_test, y_test = make_training_frame(test_df)

    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Failure rate (train): {y_train.mean():.2%}")

    print("\nTraining Random Forest baseline...")
    rf = train_baseline(X_train, y_train)
    print("Training XGBoost model...")
    xgb = train_classifier(X_train, y_train)
    print("Training Isolation Forest anomaly detector...")
    iso = train_anomaly_detector(X_train)

    rf_metrics = evaluate(rf, X_test, y_test)
    xgb_metrics = evaluate(xgb, X_test, y_test)
    metrics = pd.DataFrame([rf_metrics, xgb_metrics], index=["RandomForest", "XGBoost"])
    metrics.to_csv(ARTIFACT_DIR / "metrics.csv")

    joblib.dump(xgb, MODEL_DIR / "failure_model.joblib")
    joblib.dump(rf, MODEL_DIR / "baseline_model.joblib")
    joblib.dump(iso, MODEL_DIR / "anomaly_model.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_schema.joblib")

    metadata = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "random_state": 42,
        "test_size": 0.20,
        "validation_fraction_of_train": 0.20,
        "target": "Machine failure",
        "selected_model": "XGBoost",
        "test_metrics": xgb_metrics,
        "note": "AI4I is a synthetic benchmark. Metrics are benchmark results, not guarantees of industrial performance."
    }
    (ARTIFACT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n=== Test metrics: XGBoost ===")
    for k, v in xgb_metrics.items():
        print(f"{k:24s}: {v:.4f}" if isinstance(v, float) else f"{k:24s}: {v}")
    print("\nSaved models to models/ and metrics to artifacts/.")

if __name__ == "__main__":
    main()
