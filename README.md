# MLOPS ZoomCamp 2025 - Heart Failure Prediction

## Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease. The goal is to demonstrate best practices in MLOps, including experiment tracking, workflow orchestration, model deployment, and continuous monitoring, using a real-world healthcare dataset.

## Problem Statement

Heart disease remains a leading cause of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes by enabling timely diagnosis, preventive measures, and effective treatment plans.

This project aims to develop a robust machine learning model that can **predict the presence or absence of heart disease** in patients based on various clinical and demographic attributes. This is framed as a **binary classification problem**, where the model will output a probability or a binary decision (0 for no heart disease, 1 for heart disease).

## Dataset

The project utilizes the **Heart Failure Prediction Dataset**.

* **Source:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)
* **Description:** This dataset contains 918 entries with 12 features that include patient information such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, oldpeak (ST depression induced by exercise relative to rest), ST-slope of the peak exercise ST segment, and the target variable.
* **Target Variable:** `HeartDisease` (0 = No Heart Disease, 1 = Heart Disease).
* **ML Task:** Binary Classification.

## Project Contents

This project builds an XGBoost classifier to predict heart disease based on patient data. The MLOps pipeline automates key stages:

1.  **Data Ingestion & Preprocessing:** Handles raw data loading and transformation into model-ready features.
2.  **Experiment Tracking & Hyperparameter Tuning:** Uses MLflow and Optuna to track training runs, log metrics, and find optimal model parameters.
3.  **Workflow Orchestration:** Leverages Prefect to define, run, and schedule the end-to-end training pipeline.
4.  **Model Registry:** MLflow Model Registry is used to manage and version registered models.
5.  **Model Deployment:** The best model is served via a FastAPI REST API, containerized using Docker.
6.  **Monitoring & Observability:** Prometheus collects service metrics from the FastAPI app, visualized in Grafana.
7.  **Data Drift Detection:** Evidently AI is integrated to detect shifts in production data compared to training data.

## 🚀 Getting Started

Follow these steps to set up and run the entire MLOps pipeline locally.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Git:** For cloning the repository.
* **Python 3.12.1** (or compatible version): The project was developed with this version.
* **pip:** Python package installer.
* **Docker Desktop (or Docker Engine on Linux):** For containerization and running services.

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/nicoalpis/heart-failure-prediction.git
````

### 2. Prepare Data

Ensure `heart.csv` dataset is located in the root directory of your cloned repository.

### 3. Set up Python Environment & Install Dependencies

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start MLflow Tracking Server

The MLflow Tracking Server is where all the experiment runs, metrics, and model versions are stored. It will also serve artifacts locally.

Open a **new terminal window** and run:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

  * Keep this terminal running.
  * You can access the MLflow UI at `http://127.0.0.1:5000`.

### 5. Start Prefect Server

Prefect orchestrates the training pipeline.

```bash
prefect server start
```
  * Keep this terminal running.
  * You can access the Prefect UI at `http://127.0.0.1:4200`.

### 6. Run the Training Workflow

Now, let's train the model and register it with MLflow.

From the project's root directory (where `train_flow.py` is), run:

```bash
python train_flow.py
```

  * This will run the Prefect flow locally, perform hyperparameter tuning, train an XGBoost model, and register the best model in the MLflow Model Registry.
  * Verify in MLflow UI (`http://127.0.0.1:5000`) that an experiment run is logged and a model (`HeartDiseaseXGBoostModel`) is registered under "Models".

### 7. Build and Run Docker Compose Stack (FastAPI, Prometheus, Grafana)

We'll use Docker Compose to run the prediction API, Prometheus (for metrics collection), and Grafana (for dashboards).

**Important:** Before proceeding, you need to configure `docker-compose.yml`:

1.  **Open `docker-compose.yml`** in project's root.
2.  **Adjust the `volumes` path for the `fastapi-app` service:**
    Change `- "/workspaces/heart-failure-prediction:/workspaces/heart-failure-prediction"` to the **actual absolute path of your project's root directory on your host machine.**
3.  **Verify `MLFLOW_TRACKING_URI` for `fastapi-app`:** Ensure `MLFLOW_TRACKING_URI=http://172.17.0.1:5000` matches the host and port where your MLflow UI is running.

**Now, run Docker Compose:**

From your project's root directory in a terminal:

```bash
docker-compose up --build -d
```

  * `--build`: Ensures the FastAPI Docker image is built with the latest changes (including `prometheus_client`).
  * `-d`: Runs the services in detached mode (in the background).

### 8. Access the Services

Once Docker Compose is up, you can access the following in your web browser:

  * **FastAPI Prediction API:** `http://127.0.0.1:8000/docs` (interactive API documentation)
  * **Prometheus UI:** `http://localhost:9090`
  * **Grafana UI:** `http://localhost:3000` (Default login: `admin`/`admin`)

### 9. Test Prediction API & Metrics

1.  Go to `http://127.0.0.1:8000/docs`.
2.  Click on `/predict/` -\> "Try it out".
3.  Modify the example request body if you wish, and click "Execute". You should get a prediction response.
4.  Go to `http://127.0.0.1:8000/metrics`. You should see Prometheus metrics, and after making predictions, `http_requests_total` and `model_predictions_total` should increment.

### 10. Configure Grafana Dashboard

1.  Log in to Grafana (`http://localhost:3000`) with `admin`/`admin`.
2.  **Add Prometheus Data Source:**
      * Click the **Gear icon (Configuration)** on the left sidebar -\> **Data sources**.
      * Click **"Add data source"**.
      * Select **"Prometheus"**.
      * In the **HTTP** section, set the **URL** to `http://prometheus:9090`.
      * Click **"Save & test"**.
3.  **Create a Dashboard:**
      * Click the **" +" icon (Create)** on the left sidebar -\> **Dashboard**.
      * Click **"Add a new panel"**.
      * In the "Query" tab, select your Prometheus data source.
      * Try queries like:
          * `rate(http_requests_total[5m])` for API request rate.
          * `rate(model_predictions_total[5m])` for model prediction rate.
          * `http_request_duration_seconds_sum / http_request_duration_seconds_count` for average latency.
      * Explore different visualizations.

## 🧹 Cleaning Up

To stop and remove all Docker Compose services, networks, and volumes:

```bash
docker-compose down -v
```

To stop MLflow and Prefect Server, go to their respective terminals and press `Ctrl+C`.

## 🧑‍💻 Project Structure

```
.
├── mlruns/                          # MLflow local tracking server data (experiments, artifacts, models)
├── heart.csv                        # Raw dataset
├── Dockerfile                       # Defines the Docker image for the FastAPI app
├── docker-compose.yml               # Orchestrates FastAPI, Prometheus, Grafana services
├── prometheus.yml                   # Prometheus configuration for scraping metrics
├── predict.py                       # FastAPI application for model inference and metrics
├── train.py                         # Python script for data preprocessing, training, and MLflow logging
├── train_flow.py                    # Prefect flow for data preprocessing, training, and MLflow logging
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Technologies Used

  * **Python 3.12.1**
  * **XGBoost:** Machine learning model.
  * **Pandas, Scikit-learn:** Data manipulation and preprocessing.
  * **MLflow:** Experiment tracking, model registry, artifact management.
  * **Prefect:** Workflow orchestration.
  * **FastAPI:** REST API framework for model serving.
  * **Uvicorn:** ASGI server for FastAPI.
  * **Docker:** Containerization of the inference service.
  * **Docker Compose:** Orchestration of multiple Docker containers (FastAPI, Prometheus, Grafana).
  * **Prometheus:** Time-series monitoring system for collecting metrics.
  * **Grafana:** Data visualization and dashboarding tool.

-----

```
```