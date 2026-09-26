from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from .model import make_training_frame

MODEL = Path("models/failure_model.joblib")
ANOMALY = Path("models/anomaly_model.joblib")


def load_models():
    if not MODEL.exists() or not ANOMALY.exists():
        raise FileNotFoundError("Models not found. Run: python train.py")
    return joblib.load(MODEL), joblib.load(ANOMALY)


def health_band(prob: float, anomaly: bool) -> str:
    if prob >= 0.70 or (anomaly and prob >= 0.35):
        return "Critical"
    if prob >= 0.35 or anomaly:
        return "Warning"
    return "Normal"


def recommendation(prob: float, anomaly: bool) -> str:
    if prob >= 0.70 and anomaly:
        return "Immediate maintenance inspection recommended."
    if prob >= 0.70:
        return "Prioritize maintenance inspection before the next operating cycle."
    if anomaly:
        return "Inspect sensor readings and machine condition; abnormal behavior detected."
    if prob >= 0.35:
        return "Schedule preventive inspection and monitor the next operating cycles."
    return "Continue normal monitoring."


def predict_row(row: dict):
    model, anomaly_model = load_models()
    df = pd.DataFrame([row])
    X, _ = make_training_frame(pd.concat([df, pd.DataFrame({"Machine failure":[0]})], axis=1))
    p = float(model.predict_proba(X)[:, 1][0])
    anomaly_raw = int(anomaly_model.predict(X)[0]) == -1
    return {
        "failure_probability": p,
        "failure_prediction": int(p >= 0.50),
        "anomaly_detected": anomaly_raw,
        "health_status": health_band(p, anomaly_raw),
        "maintenance_recommendation": recommendation(p, anomaly_raw),
    }
