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

## Technologies Used (Planned)

This project will leverage a suite of modern MLOps tools and cloud services to build a comprehensive pipeline:

* **Programming Language:** Python
* **Cloud Platform:** [To be decided/specified, e.g., AWS, GCP, or Azure]
* **Experiment Tracking & Model Registry:** MLflow
* **Workflow Orchestration:** Prefect
* **Containerization:** Docker
* **Model Deployment:** [e.g., FastAPI + Docker + Cloud Run/ECS]
* **Model Monitoring:** Evidently AI
* **Infrastructure as Code (IaC):** Terraform
* **CI/CD:** GitHub Actions
* **Code Quality:** Black, Flake8, pre-commit hooks
* **Automation:** Makefile

## MLOps Lifecycle Overview

The project will cover the following key stages of the MLOps lifecycle:

1.  **Experiment Tracking:** Using MLflow to log parameters, metrics, and artifacts for each model training run, ensuring reproducibility and easy comparison of experiments.
2.  **Model Training Pipeline:** Automating the data preprocessing, model training, and evaluation steps using Prefect flows.
3.  **Model Deployment:** Deploying the trained model as a web service for real-time inference, containerized with Docker and hosted on a cloud platform.
4.  **Model Monitoring:** Implementing a monitoring solution (Evidently AI) to track model performance, data drift, and concept drift in production.
5.  **Best Practices & CI/CD:** Adhering to software engineering best practices including unit/integration testing, code linting, and continuous integration/delivery pipelines using GitHub Actions.
6.  **Infrastructure as Code:** Managing cloud infrastructure setup using Terraform for reproducibility and version control.

---

**Next Steps:**

We'll continue to build out the project step-by-step, and this `README.md` will be updated with detailed instructions on setup, running the pipeline, and accessing deployed services.
