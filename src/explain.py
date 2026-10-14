
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def top_shap_features(pipeline, X: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Return the strongest feature contributions for one prediction."""

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = pre.transform(X)
    names = pre.get_feature_names_out()

    # Use XGBoost's native SHAP contributions to avoid the
    # TreeExplainer base_score parsing error.
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        matrix = xgb.DMatrix(np.asarray(transformed))
        contributions = model.get_booster().predict(
            matrix,
            pred_contribs=True,
        )

        values = np.asarray(contributions)
        row_values = values[0, :-1]

    else:
        # Keep SHAP TreeExplainer for other tree-based models.
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)

        if isinstance(values, list):
            values = values[1]

        values = np.asarray(values)

        if values.ndim == 3:
            values = values[0, :, -1]

        row_values = values[0]

    result = pd.DataFrame({
        "Feature": names,
        "SHAP value": row_values,
        "Absolute impact": np.abs(row_values),
    })

    return result.sort_values(
        "Absolute impact",
        ascending=False,
    ).head(top_n)
