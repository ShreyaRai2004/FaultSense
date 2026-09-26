from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

from .preprocess import TARGET, add_features, DROP_COLUMNS


def make_training_frame(df: pd.DataFrame):
    df = add_features(df)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return X, y


def make_xgb(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )


def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    return pre, numeric, categorical


def train_classifier(X_train, y_train):
    pre, numeric, categorical = build_preprocessor(X_train)
    pos = max(1, int((y_train == 0).sum()))
    neg = max(1, int((y_train == 1).sum()))
    ratio = pos / neg
    model = make_xgb(ratio)
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def train_baseline(X_train, y_train):
    pre, _, _ = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=4
    )
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def train_anomaly_detector(X_train):
    pre, _, _ = build_preprocessor(X_train)
    # Contamination is deliberately conservative because failures are rare.
    detector = IsolationForest(
        n_estimators=250, contamination=0.03, random_state=42, n_jobs=4
    )
    pipe = Pipeline([("preprocessor", pre), ("model", detector)])
    pipe.fit(X_train)
    return pipe


def save_model(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
