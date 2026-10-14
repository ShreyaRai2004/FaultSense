"""FaultSense Copilot: tool-using LLM assistant for machine-risk analysis."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.explain import top_shap_features
from src.model import make_training_frame
from src.predict import predict_row


def _machine_row(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict[str, Any]:
    return {
        "UDI": 0,
        "Product ID": "COPILOT",
        "Type": machine_type,
        "Air temperature [K]": float(air_temperature),
        "Process temperature [K]": float(process_temperature),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
    }


def _predict_with_configured_api(row: dict[str, Any]) -> dict[str, Any]:
    """Use a configured FastAPI service; otherwise use the same local predictor."""
    base_url = os.getenv("FAULTSENSE_API_URL", "").strip().rstrip("/")
    if base_url:
        endpoint = base_url if base_url.endswith("/predict") else f"{base_url}/predict"
        payload = {
            "Type": row["Type"],
            "air_temperature": row["Air temperature [K]"],
            "process_temperature": row["Process temperature [K]"],
            "rotational_speed": row["Rotational speed [rpm]"],
            "torque": row["Torque [Nm]"],
            "tool_wear": row["Tool wear [min]"],
        }
        response = requests.post(endpoint, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()

    # This calls the same prediction function that backs api.py's /predict route.
    return predict_row(row)


def build_tools(model):
    """Build tools with access to the current trained model."""

    @tool
    def predict_failure(
        machine_type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
    ) -> str:
        """Predict failure probability, anomaly status, health band, and maintenance action for the supplied machine sensor readings. Use this when the user asks about risk or machine health."""
        row = _machine_row(
            machine_type, air_temperature, process_temperature,
            rotational_speed, torque, tool_wear,
        )
        try:
            result = _predict_with_configured_api(row)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"Prediction failed: {exc}"})

    @tool
    def explain_prediction(
        machine_type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
    ) -> str:
        """Return the strongest local SHAP feature contributions for the supplied machine sensor readings. Use this when the user asks why risk is high or what factors influenced a prediction."""
        row = _machine_row(
            machine_type, air_temperature, process_temperature,
            rotational_speed, torque, tool_wear,
        )
        training_row = {
            **row,
            "Machine failure": 0,
            "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        }
        try:
            X, _ = make_training_frame(pd.DataFrame([training_row]))
            factors = top_shap_features(model, X, top_n=5)
            factors["Direction"] = factors["SHAP value"].apply(
                lambda value: "Raises failure risk" if float(value) > 0 else "Lowers failure risk"
            )
            return factors[["Feature", "SHAP value", "Direction"]].to_json(orient="records")
        except Exception as exc:
            return json.dumps({"error": f"SHAP explanation failed: {exc}"})

    return [predict_failure, explain_prediction]


def run_copilot(question: str, current_machine: dict[str, Any], model) -> str:
    """Answer a question using bounded tool calls and the current sidebar readings."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add your Gemini API key to the local .env file "
            "or Streamlit Secrets, then restart the app."
        )

    tools = build_tools(model)
    tools_by_name = {item.name: item for item in tools}
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0,
    ).bind_tools(tools)

    prompt = f"""You are FaultSense Copilot, an assistant for a predictive-maintenance demo.
Use the tools for any question about machine failure risk or the reasons behind a prediction. Do not invent model results, SHAP values, machine history, or sensor readings. Explain technical results in clear, concise plain English. Include the failure probability and health status when the prediction tool returns them; distinguish anomaly detection from failure probability. When useful, give the maintenance recommendation returned by the tool. SHAP direction is model contribution, not proof of causation. If the user asks for a specific machine ID, explain that this app evaluates the current sidebar sensor readings and does not have a machine-ID registry. This is a synthetic benchmark demonstration, not an industrial safety system.

Current sidebar sensor readings (use these exact values when calling tools):
- Machine type: {current_machine['Type']}
- Air temperature [K]: {current_machine['Air temperature [K]']}
- Process temperature [K]: {current_machine['Process temperature [K]']}
- Rotational speed [rpm]: {current_machine['Rotational speed [rpm]']}
- Torque [Nm]: {current_machine['Torque [Nm]']}
- Tool wear [min]: {current_machine['Tool wear [min]']}
"""
    messages = [SystemMessage(content=prompt), HumanMessage(content=question)]

    for _ in range(4):
        response = llm.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            return json.dumps(content, ensure_ascii=False)

        for call in response.tool_calls:
            tool_obj = tools_by_name.get(call["name"])
            if tool_obj is None:
                tool_result = json.dumps({"error": "Unknown tool requested."})
            else:
                try:
                    tool_result = tool_obj.invoke(call["args"])
                except Exception as exc:
                    tool_result = json.dumps({"error": f"Tool execution failed: {exc}"})
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))

    return "I couldn't complete the analysis in the available tool steps. Please ask a more specific question."
