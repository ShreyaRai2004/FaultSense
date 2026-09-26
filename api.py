from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict_row

app = FastAPI(title="FaultSense API", version="1.0.0")

class MachineInput(BaseModel):
    Type: str = Field(pattern="^[LMH]$")
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def root():
    return {"service": "FaultSense", "status": "running"}

@app.post("/predict")
def predict(data: MachineInput):
    row = {
        "UDI": 0,
        "Product ID": "API",
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "Machine failure": 0, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }
    return predict_row(row)
