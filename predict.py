import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import xgboost as xgb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using an XGBoost model from MLflow.",
    version="1.0.0"
)

# --- MLflow Model Loading ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = "HeartDiseaseXGBoostModel"

# Set MLflow tracking URI explicitly
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Variable to hold the loaded model
model = None

@app.on_event("startup")
async def load_model():
    """
    Load the latest version of the registered MLflow model when the FastAPI app starts up.
    """
    global model
    logger.info(f"Loading model '{MLFLOW_MODEL_NAME}' from MLflow Model Registry at {MLFLOW_TRACKING_URI}...")
    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.xgboost.load_model(model_uri)
        logger.info(f"Model '{MLFLOW_MODEL_NAME}' loaded successfully from {model_uri}.")
    except Exception as e:
        logger.error(f"Failed to load model '{MLFLOW_MODEL_NAME}': {e}")
        model = None
        # In a production scenario, you might want to raise an exception here
        # to prevent the server from starting with no model loaded.

# --- Data Model for Request Body ---
# CORRECTED: Updated to match the exact columns from X_train
class PatientData(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex_M: float # 0.0 for Female, 1.0 for Male
    ChestPainType_ATA: float # 0.0 or 1.0
    ChestPainType_NAP: float # 0.0 or 1.0
    ChestPainType_TA: float # 0.0 or 1.0 (Previously ChestPainType_ASY)
    RestingECG_Normal: float # 0.0 or 1.0
    RestingECG_ST: float # 0.0 or 1.0
    ExerciseAngina_Y: float # 0.0 for No, 1.0 for Yes
    ST_Slope_Flat: float # 0.0 or 1.0
    ST_Slope_Up: float # 0.0 or 1.0

# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict_heart_disease(data: PatientData):
    """
    Receives patient data, makes a prediction using the loaded XGBoost model,
    and returns the predicted probability and class.
    """
    if model is None:
        logger.error("Model is not loaded. Cannot make predictions.")
        return {"error": "Model not loaded, please check server logs."}

    input_df = pd.DataFrame([data.dict()])
    
    # CORRECTED: Updated to match the exact order and names from X_train
    expected_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    # Reindex input_df to match the expected column order, filling missing with 0
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Make prediction
    dmatrix_input = xgb.DMatrix(input_df)
    prediction_proba = model.predict(dmatrix_input)[0]
    prediction_class = 1 if prediction_proba >= 0.5 else 0

    logger.info(f"Prediction for input data: Probability={prediction_proba:.4f}, Class={prediction_class}")

    return {
        "prediction_probability": float(prediction_proba),
        "predicted_class": int(prediction_class),
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": "latest"
    }

# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/health/")
async def health_check():
    """Returns a simple message to indicate the API is running."""
    if model:
        return {"status": "ok", "message": "API is running and model is loaded."}
    else:
        return {"status": "error", "message": "API is running but model failed to load."}


# --- To run the API directly (for development/testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)