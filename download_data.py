from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT = Path("data/ai4i2020.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("Fetching UCI AI4I 2020 Predictive Maintenance Dataset...")
    ds = fetch_ucirepo(id=601)
    features = ds.data.features.copy()
    targets = ds.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
    print("Columns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()
