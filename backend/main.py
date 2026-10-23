from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.schemas import (
    DemandPredictionRequest,
    DemandPredictionResponse,
)

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI-Powered Retail Pricing Intelligence",
    description="API for retail demand prediction and price optimization",
    version="1.0.0",
)

# --------------------------------------------------
# Load the trained demand prediction model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Retail Pricing Intelligence API is running",
        "status": "success",
    }


# --------------------------------------------------
# Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "FastAPI backend is running",
    }


# --------------------------------------------------
# Demand prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict-demand",
    response_model=DemandPredictionResponse,
)
def predict_demand(request: DemandPredictionRequest):
    try:
        # Convert the incoming request into a DataFrame
        input_data = pd.DataFrame(
            [request.model_dump(by_alias=True)]
        )

        # Convert Promotion values to numeric values
        input_data["Promotion"] = input_data["Promotion"].replace({
            "Yes": 1,
            "No": 0,
        })

        # Convert Epidemic values to numeric values
        input_data["Epidemic"] = input_data["Epidemic"].replace({
            "Yes": 1,
            "No": 0,
        })

        # Predict demand using the trained model
        predicted_demand = model.predict(input_data)

        # Return the prediction as a JSON response
        return DemandPredictionResponse(
            predicted_demand=float(predicted_demand[0])
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Demand prediction failed: {str(exc)}",
        ) from exc