import pandas as pd
from src.preprocess import clean_columns, add_features

def test_ai4i_schema_is_normalized():
    df = pd.DataFrame([{
        "air_temperature_k": 300,
        "process_temperature_k": 310,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40,
        "tool_wear_min": 100,
        "type": "M",
        "machine_failure": 0,
    }])
    cleaned = clean_columns(df)
    assert "Process temperature [K]" in cleaned.columns
    enriched = add_features(df)
    assert "Temperature_Delta" in enriched.columns
    assert "Wear_Load_Index" in enriched.columns
