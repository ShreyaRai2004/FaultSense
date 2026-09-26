from __future__ import annotations
import numpy as np
import pandas as pd
import shap


def top_shap_features(pipeline, X: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Return the strongest local SHAP contributors for one prediction."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = pre.transform(X)
    names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    row_values = values[0]
    result = pd.DataFrame({
        "Feature": names,
        "SHAP value": row_values,
        "Absolute impact": np.abs(row_values),
    }).sort_values("Absolute impact", ascending=False).head(top_n)
    return result
