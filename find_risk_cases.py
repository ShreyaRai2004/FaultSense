from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import clean_columns
from src.model import make_training_frame


DATA = Path("data/ai4i2020.csv")
MODEL = Path("models/failure_model.joblib")


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python download_data.py"
        )

    if not MODEL.exists():
        raise FileNotFoundError(
            "Model not found. Run: python train.py"
        )

    df = pd.read_csv(DATA)
    df = clean_columns(df)

    model = joblib.load(MODEL)

    X, y = make_training_frame(df)

    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["Failure Probability"] = probabilities

    print("\n========== LOW RISK EXAMPLE ==========")

    low = result[result["Failure Probability"] < 0.05]

    if len(low) > 0:
        row = low.iloc[0]
        print_row(row)
    else:
        print("No low-risk example found.")

    print("\n========== MODERATE RISK EXAMPLE ==========")

    moderate = result[
        (result["Failure Probability"] >= 0.35)
        & (result["Failure Probability"] < 0.70)
    ].copy()

    if len(moderate) > 0:
        moderate["Distance"] = abs(
            moderate["Failure Probability"] - 0.50
        )
        row = moderate.sort_values("Distance").iloc[0]
        print_row(row)
    else:
        print("No moderate-risk example found.")

    print("\n========== HIGH RISK EXAMPLE ==========")

    high = result[result["Failure Probability"] >= 0.70]

    if len(high) > 0:
        high = high.sort_values(
            "Failure Probability",
            ascending=False
        )
        row = high.iloc[0]
        print_row(row)
    else:
        print("No high-risk example found.")


def print_row(row):
    print(f"Type: {row['Type']}")
    print(f"Air Temperature: {row['Air temperature [K]']}")
    print(f"Process Temperature: {row['Process temperature [K]']}")
    print(f"Rotational Speed: {row['Rotational speed [rpm]']}")
    print(f"Torque: {row['Torque [Nm]']}")
    print(f"Tool Wear: {row['Tool wear [min]']}")
    print(f"Actual Machine Failure: {row['Machine failure']}")
    print(
        f"Model Failure Risk: "
        f"{row['Failure Probability']:.2%}"
    )


if __name__ == "__main__":
    main()