from __future__ import annotations

import re
import numpy as np
import pandas as pd

TARGET = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID", TARGET, "TWF", "HDF", "PWF", "OSF", "RNF"]

# The UCI dataset has stable semantic column names, but CSV exports can differ
# in whitespace/case/punctuation. We canonicalize aliases instead of requiring
# the user to manually edit the downloaded data.
ALIASES = {
    "UDI": ["udi"],
    "Product ID": ["productid", "product_id", "product id"],
    "Type": ["type", "machine_type", "machine type"],
    "Air temperature [K]": [
        "airtemperaturek", "air temperature k", "air_temperature_k",
        "airtemperature", "air temperature"
    ],
    "Process temperature [K]": [
        "processtemperaturek", "process temperature k", "process_temperature_k",
        "processtemperature", "process temperature"
    ],
    "Rotational speed [rpm]": [
        "rotationalspeedrpm", "rotational speed rpm", "rotational_speed_rpm",
        "rotationalspeed", "rotational speed"
    ],
    "Torque [Nm]": [
        "torquenm", "torque nm", "torque_nm", "torque"
    ],
    "Tool wear [min]": [
        "toolwearmin", "tool wear min", "tool_wear_min",
        "toolwear", "tool wear"
    ],
    "Machine failure": [
        "machinefailure", "machine failure", "machine_failure", "failure"
    ],
    "TWF": ["twf"],
    "HDF": ["hdf"],
    "PWF": ["pwf"],
    "OSF": ["osf"],
    "RNF": ["rnf"],
}


def _norm(name: object) -> str:
    s = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV column variants to the canonical AI4I names."""
    out = df.copy()
    lookup = {_norm(c): c for c in out.columns}
    rename = {}

    for canonical, aliases in ALIASES.items():
        candidates = [canonical] + aliases
        for candidate in candidates:
            key = _norm(candidate)
            if key in lookup:
                rename[lookup[key]] = canonical
                break

    out = out.rename(columns=rename)

    required = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            "FaultSense could not recognize the required dataset columns: "
            + ", ".join(missing)
            + ". Found columns: "
            + ", ".join(map(str, out.columns))
        )

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_columns(df)

    # Features describe operating conditions rather than identifiers.
    out["Temperature_Delta"] = (
        out["Process temperature [K]"] - out["Air temperature [K]"]
    )
    out["Power_Proxy"] = (
        out["Torque [Nm]"] * out["Rotational speed [rpm]"]
    )
    out["Torque_Speed_Ratio"] = (
        out["Torque [Nm]"] / (out["Rotational speed [rpm]"] + 1e-6)
    )
    out["Thermal_Load_Index"] = (
        out["Process temperature [K]"] * out["Torque [Nm]"]
    )
    out["Wear_Load_Index"] = (
        out["Tool wear [min]"] * out["Torque [Nm]"]
    )
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLUMNS]


def build_xy(df: pd.DataFrame):
    df = add_features(df)
    y = df[TARGET].astype(int)
    X = df[feature_columns(df)].copy()
    X = pd.get_dummies(X, columns=["Type"], drop_first=False)
    return X, y
