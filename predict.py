import os
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pandas as pd
import mlflow
import xgboost as xgb
import logging
import time

# Import Prometheus client libraries
from prometheus_client import generate_latest, Counter, Histogram

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using an XGBoost model from MLflow.",
    version="1.0.0"
)

# --- Prometheus Metrics ---
REQUESTS_TOTAL = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP request latency in seconds', ['method', 'endpoint']
)
PREDICTION_COUNT = Counter(
    'model_predictions_total', 'Total model predictions made'
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds', 'Time taken for model prediction in seconds'
)

# --- MLflow Model Loading ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.0.1:5000") # Ensure this matches your MLflow UI address
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

# --- Middleware for request latency and count ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    method = request.method
    status_code = response.status_code

    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)

    return response

# --- Data Model for Request Body ---
class PatientData(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex_M: float
    ChestPainType_ATA: float
    ChestPainType_NAP: float
    ChestPainType_TA: float
    RestingECG_Normal: float
    RestingECG_ST: float
    ExerciseAngina_Y: float
    ST_Slope_Flat: float
    ST_Slope_Up: float

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

    expected_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Measure prediction latency and count
    pred_start_time = time.time()
    dmatrix_input = xgb.DMatrix(input_df)
    prediction_proba = model.predict(dmatrix_input)[0]
    pred_process_time = time.time() - pred_start_time

    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(pred_process_time)

    prediction_class = 1 if prediction_proba >= 0.5 else 0

    logger.info(f"Prediction for input data: Probability={prediction_proba:.4f}, Class={prediction_class}")

    return {
        "prediction_probability": float(prediction_proba),
        "predicted_class": int(prediction_class),
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": "latest"
    }

# --- Health Check Endpoint ---
@app.get("/health/")
async def health_check():
    """Returns a simple message to indicate the API is running."""
    if model:
        return {"status": "ok", "message": "API is running and model is loaded."}
    else:
        return {"status": "error", "message": "API is running but model failed to load."}

# --- Metrics Endpoint for Prometheus ---
@app.get("/metrics")
async def metrics():
    """
    Exposes Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type="text/plain")

# --- To run the API directly (for development/testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)