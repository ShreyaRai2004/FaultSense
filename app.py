from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.model import make_training_frame
from src.predict import health_band, recommendation
from src.explain import top_shap_features

st.set_page_config(page_title="FaultSense", page_icon="⚙️", layout="wide")

MODEL_PATH = Path("models/failure_model.joblib")
ANOMALY_PATH = Path("models/anomaly_model.joblib")


def prepare_models():
    """Download the dataset and train models when running on a fresh deployment."""
    if MODEL_PATH.exists() and ANOMALY_PATH.exists():
        return

    st.info("Preparing the machine-learning models for the first run...")
    import download_data
    import train

    data_path = Path("data/ai4i2020.csv")
    if not data_path.exists():
        with st.spinner("Downloading the UCI AI4I 2020 dataset..."):
            download_data.main()
    with st.spinner("Training FaultSense models..."):
        train.main()


st.title("⚙️ FaultSense")
st.caption("Predictive Maintenance & Machine Health Intelligence")
st.markdown(
    "Predict machine failure risk from operating conditions, detect unusual patterns, "
    "and translate model output into a maintenance action."
)

prepare_models()
model = joblib.load(MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

with st.sidebar:
    st.header("Machine Inputs")
    product_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
    air = st.number_input("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
    process = st.number_input("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
    speed = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500, 1)
    torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0, 0.1)
    wear = st.number_input("Tool Wear [min]", 0, 300, 100, 1)
    run = st.button("Assess Machine Health", type="primary", use_container_width=True)

current_row = {
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
}


dashboard_tab, copilot_tab = st.tabs(["📊 Machine Dashboard", "💬 Ask FaultSense"])

with dashboard_tab:
    st.subheader("Machine Health Assessment")
    if run:
        row_df = pd.DataFrame([current_row])
        X, _ = make_training_frame(row_df)
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
        shap_df["Direction"] = np.where(
            shap_df["SHAP value"] >= 0, "Raises failure risk", "Lowers failure risk"
        )
        st.dataframe(
            shap_df[["Feature", "SHAP value", "Direction"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Derived Operating Indicators")
        indicators = pd.DataFrame({
            "Indicator": ["Temperature Delta", "Power Proxy", "Torque/Speed Ratio", "Thermal Load", "Wear Load"],
            "Value": [process - air, torque * speed, torque / (speed + 1e-6), process * torque, wear * torque],
        })
        st.dataframe(indicators, use_container_width=True, hide_index=True)
    else:
        st.info("Enter machine operating conditions in the sidebar and click **Assess Machine Health**.")

    st.caption(
        "FaultSense is a benchmark ML demonstration using the UCI AI4I 2020 synthetic dataset; "
        "outputs should not be treated as real industrial safety decisions."
    )

with copilot_tab:
    st.subheader("Ask FaultSense")
    st.write(
        "Ask about the current sensor readings, failure risk, SHAP factors, or the next maintenance action. "
        "The assistant uses the existing FaultSense prediction and explanation logic."
    )
    with st.expander("Current readings the Copilot will analyze"):
        st.json({
            "Machine type": product_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        })

    if not os.getenv("GOOGLE_API_KEY", "").strip():
        st.warning(
            "Copilot needs a Gemini API key. Add GOOGLE_API_KEY to your local .env file "
            "or to Streamlit Cloud → App settings → Secrets. Do not paste your key into chat or commit it to GitHub."
        )

    if "faultsense_chat" not in st.session_state:
        st.session_state.faultsense_chat = []

    for message in st.session_state.faultsense_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("e.g. Why is this machine at risk?", key="faultsense_chat_input")
    if question:
        st.session_state.faultsense_chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking machine risk and relevant factors..."):
                try:
                    from agent import run_copilot
                    copilot_row = {
                        "Type": product_type,
                        "Air temperature [K]": air,
                        "Process temperature [K]": process,
                        "Rotational speed [rpm]": speed,
                        "Torque [Nm]": torque,
                        "Tool wear [min]": wear,
                    }
                    answer = run_copilot(question, copilot_row, model)
                except Exception as exc:
                    answer = f"I couldn't run the Copilot yet: {exc}"
                st.markdown(answer)
        st.session_state.faultsense_chat.append({"role": "assistant", "content": answer})

    if st.button("Clear chat history", key="clear_faultsense_chat"):
        st.session_state.faultsense_chat = []
        st.rerun()
