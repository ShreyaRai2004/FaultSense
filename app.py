from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.model import make_training_frame
from src.predict import health_band, recommendation
from src.explain import top_shap_features

st.set_page_config(page_title="FaultSense", page_icon="⚙️", layout="wide")

MODEL_PATH = Path("models/failure_model.joblib")
ANOMALY_PATH = Path("models/anomaly_model.joblib")

st.title("⚙️ FaultSense")
st.caption("Predictive Maintenance & Machine Health Intelligence")

if not MODEL_PATH.exists() or not ANOMALY_PATH.exists():
    st.warning("Trained models are not available yet. Run `python download_data.py` and `python train.py` locally first.")
    st.stop()

model = joblib.load(MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

st.markdown("Predict machine failure risk from operating conditions, detect unusual patterns, and translate the model output into a maintenance action.")

with st.sidebar:
    st.header("Machine Inputs")
    product_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
    air = st.number_input("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
    process = st.number_input("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
    speed = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500, 10)
    torque = st.number_input("Torque [Nm]", 5.0, 80.0, 40.0, 0.5)
    wear = st.number_input("Tool Wear [min]", 0, 300, 100, 1)
    run = st.button("Assess Machine Health", type="primary", use_container_width=True)

if run:
    row = pd.DataFrame([{
        "UDI": 0,
        "Product ID": "APP",
        "Type": product_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Machine failure": 0,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }])
    X, _ = make_training_frame(row)
    probability = float(model.predict_proba(X)[:, 1][0])
    anomaly = int(anomaly_model.predict(X)[0]) == -1
    status = health_band(probability, anomaly)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure Risk", f"{probability:.2%}")
    c2.metric("Machine Health", status)
    c3.metric("Anomaly", "Detected" if anomaly else "Not detected")

    st.subheader("Maintenance Recommendation")
    st.info(recommendation(probability, anomaly))

    st.subheader("Why did the model make this prediction?")
    shap_df = top_shap_features(model, X)
    shap_df["Direction"] = np.where(shap_df["SHAP value"] >= 0, "Raises failure risk", "Lowers failure risk")
    st.dataframe(shap_df[["Feature", "SHAP value", "Direction"]], use_container_width=True, hide_index=True)

    st.subheader("Derived Operating Indicators")
    indicators = pd.DataFrame({
        "Indicator": ["Temperature Delta", "Power Proxy", "Torque/Speed Ratio", "Thermal Load", "Wear Load"],
        "Value": [
            process-air,
            torque*speed,
            torque/(speed+1e-6),
            process*torque,
            wear*torque,
        ]
    })
    st.dataframe(indicators, use_container_width=True, hide_index=True)

    st.caption("The model is a benchmark ML demonstration using the UCI AI4I 2020 synthetic dataset; outputs should not be treated as real industrial safety decisions.")
else:
    st.info("Enter machine operating conditions in the sidebar and click **Assess Machine Health**.")
