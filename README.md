# FaultSense — Predictive Maintenance & Machine Health Intelligence

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

FaultSense is an end-to-end machine learning application for predicting machine failure risk from operational sensor data.

The system combines data preprocessing, feature engineering, XGBoost classification, anomaly detection, SHAP explainability, machine health assessment, maintenance recommendations, FastAPI REST APIs, and an interactive Streamlit dashboard.

## 🚀 Live Demo

[Open FaultSense Live Demo](https://faultsense-6qjtnlhbg3xxvkh8jcahrq.streamlit.app/)

## ✨ Key Features

- Machine failure prediction using XGBoost
- Feature engineering from machine sensor measurements
- Random Forest baseline model comparison
- Isolation Forest anomaly detection
- SHAP-based model explainability
- Normal, Warning, and Critical machine-health classification
- Automated maintenance recommendations
- FastAPI REST API for predictions
- Interactive Streamlit dashboard
- Data validation and preprocessing pipeline

## 🧠 Machine Learning Workflow

```text
Sensor Data
     ↓
Data Cleaning & Validation
     ↓
Feature Engineering
     ↓
Train / Validation / Test Split
     ↓
Random Forest Baseline
     ↓
XGBoost Failure Prediction
     ↓
Failure Probability
     ↓
Isolation Forest Anomaly Detection
     ↓
SHAP Explainability
     ↓
Machine Health Assessment
     ↓
Maintenance Recommendation
```
## 🤖 Model Performance

XGBoost is used as the primary machine-failure prediction model, with Random Forest used as a baseline.

| Metric | XGBoost |
|---|---:|
| Accuracy | 98.90% |
| Precision | 84.85% |
| Recall | 82.35% |
| F1 Score | 83.58% |
| ROC-AUC | 0.9719 |
| PR-AUC | 0.8794 |

Because machine failures are relatively rare, the model is evaluated using precision, recall, F1, ROC-AUC, and PR-AUC in addition to accuracy.

## 🔍 Explainable AI

FaultSense uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

Instead of only displaying a failure probability, the application identifies the sensor and engineered features contributing to the prediction, making the model output easier to interpret.

## 🚨 Machine Health Assessment

The predicted failure probability and anomaly status are combined to classify machine condition:

| Status | Description |
|---|---|
| 🟢 **Normal** | Low failure risk |
| 🟡 **Warning** | Elevated failure risk or detected anomaly |
| 🔴 **Critical** | High failure risk |

The system then generates a maintenance recommendation based on the predicted risk and machine condition.

## 🛠️ Technology Stack

**Python** • **Pandas** • **NumPy** • **Scikit-learn** • **XGBoost** • **SHAP** • **FastAPI** • **Streamlit** • **Matplotlib** • **Joblib** • **Pytest** • **Git**

## 💡 Key Implementation Highlights

- Built a complete machine-learning pipeline from preprocessing to prediction.
- Engineered operational features from temperature, rotational speed, torque, and tool-wear measurements.
- Used XGBoost for nonlinear machine-failure classification.
- Evaluated the model using metrics suitable for imbalanced classification.
- Integrated Isolation Forest for detecting abnormal operating conditions.
- Implemented SHAP for model-level and prediction-level explainability.
- Connected the ML pipeline with FastAPI for REST-based prediction access.
- Built an interactive Streamlit interface for machine-health analysis.
- Added rule-based maintenance recommendations based on model output.

## 📌 Project Outcome

FaultSense demonstrates how machine-learning predictions can be transformed into an interpretable decision-support workflow:

**Sensor Data → ML Prediction → Anomaly Detection → Explanation → Health Status → Maintenance Recommendation**

##  Author

**Shreya S Rai**

MCA Student 
