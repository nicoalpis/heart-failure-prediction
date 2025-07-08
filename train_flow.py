import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from functools import partial

# Import Prefect decorators
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact # For adding info to Prefect UI

# Configure logging (same as before)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI (optional, but good practice if not using default local)
# mlflow.set_tracking_uri("http://localhost:5000") 

# Set experiment name (this will be done by MLflow within the Prefect task)
# mlflow.set_experiment("Heart Disease Prediction XGBoost Hyperopt") 

@task(log_prints=True) # log_prints=True makes print statements show in Prefect logs
def load_and_preprocess_data(file_path: str, random_state: int = 42):
    """Loads and preprocesses the heart disease dataset."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Please ensure the file is in the correct directory.")
        raise

    df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    logger.info("Categorical features one-hot encoded.")

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    return X_train, X_test, y_train, y_test

@task(log_prints=True)
def objective(params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Objective function for Hyperopt. Trains an XGBoost model and logs metrics to MLflow.
    Hyperopt minimizes the 'loss' value. For classification, we use (1 - ROC AUC).
    This function *does not* log the model artifact.
    """
    with mlflow.start_run(): # Each trial is a top-level run, no nesting
        mlflow.set_experiment("Heart Disease Prediction XGBoost Hyperopt") # Set experiment for each run
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("tuning_method", "hyperopt_trial") 
        
        train_params = params.copy()
        if 'max_depth' in train_params:
            train_params['max_depth'] = int(train_params['max_depth'])
        
        mlflow.log_params(train_params)
        logger.info(f"MLflow trial started for params: {train_params}")

        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            model = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=100, 
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                verbose_eval=False,
                early_stopping_rounds=10
            )
            logger.info("XGBoost model trained for trial.")

            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("best_iteration", model.best_iteration)
            logger.info(f"Trial metrics: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}, ROC AUC={roc_auc:.4f}")

            loss = 1 - roc_auc

            return {'loss': loss, 'status': STATUS_OK, 'roc_auc': roc_auc}

        except Exception as e:
            logger.error(f"Error during training or evaluation for trial: {e}")

@task(log_prints=True)
def find_best_run_and_params(experiment_name: str):
    """Finds the best MLflow run (hyperopt trial) and extracts its parameters."""
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    
    logger.info("Searching for the best hyperopt trial within the experiment.")

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.roc_auc DESC"], 
        max_results=1,
        filter_string="attributes.status = 'FINISHED' AND tags.\"tuning_method\" = 'hyperopt_trial'"
    )
    
    if not runs:
        logger.error(f"No successful hyperopt trials found within experiment {experiment_id}. Ensure trials are finishing with status 'FINISHED' and tag 'tuning_method' as 'hyperopt_trial'.")
        raise ValueError("No best run found.")

    best_hyperopt_run = runs[0]
    best_params_from_run = {k: v for k, v in best_hyperopt_run.data.params.items()}
    
    if 'max_depth' in best_params_from_run:
        best_params_from_run['max_depth'] = int(best_params_from_run['max_depth'])
    
    logger.info(f"Best Hyperopt trial (based on ROC AUC): run_id={best_hyperopt_run.info.run_id}, ROC AUC={best_hyperopt_run.data.metrics['roc_auc']:.4f}")
    logger.info(f"Best parameters for final model training: {best_params_from_run}")
    
    # Create a Prefect artifact summarizing the best run
    markdown_content = f"""
### Best Hyperopt Trial
- **Run ID:** `{best_hyperopt_run.info.run_id}`
- **ROC AUC:** `{best_hyperopt_run.data.metrics['roc_auc']:.4f}`
- **Parameters:**
{", ".join([f"`{k}: {v}`" for k,v in best_params_from_run.items()])}
"""
    create_markdown_artifact(
        key="best-hyperopt-run-summary",
        markdown=markdown_content,
        description="Summary of the best Hyperopt trial."
    )

    return best_params_from_run, best_hyperopt_run.info.run_id

@task(log_prints=True)
def train_and_register_best_model(
    best_params: dict, 
    X_train: pd.DataFrame, y_train: pd.Series, 
    X_test: pd.DataFrame, y_test: pd.Series,
    best_hyperopt_run_id: str,
    registered_model_name: str = "HeartDiseaseXGBoostModel"
):
    """Trains the final best model and registers it to MLflow."""
    logger.info("Training and logging the final best model.")
    # Changed: Removed 'nested=True'. This is now a top-level run.
    with mlflow.start_run(run_name="Final_Best_XGBoost_Model"):
        mlflow.set_experiment("Heart Disease Prediction XGBoost Hyperopt") # Ensure experiment is set
        mlflow.set_tag("mlops_step", "final_model_training")
        mlflow.log_params(best_params)

        dtrain_final = xgb.DMatrix(X_train, label=y_train)
        dtest_final = xgb.DMatrix(X_test, label=y_test)

        final_model = xgb.train(
            params=best_params,
            dtrain=dtrain_final,
            num_boost_round=100,
            evals=[(dtrain_final, 'train'), (dtest_final, 'eval')],
            verbose_eval=False,
            early_stopping_rounds=10
        )
        logger.info("Final best XGBoost model trained.")

        y_pred_proba_final = final_model.predict(dtest_final)
        y_pred_final = (y_pred_proba_final > 0.5).astype(int)

        final_accuracy = accuracy_score(y_test, y_pred_final)
        final_precision = precision_score(y_test, y_pred_final, zero_division=0)
        final_recall = recall_score(y_test, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_test, y_pred_final, zero_division=0)
        final_roc_auc = roc_auc_score(y_test, y_pred_proba_final)

        mlflow.log_metric("final_accuracy", final_accuracy)
        mlflow.log_metric("final_precision", final_precision)
        mlflow.log_metric("final_recall", final_recall)
        mlflow.log_metric("final_f1_score", final_f1)
        mlflow.log_metric("final_roc_auc", final_roc_auc)
        mlflow.log_metric("final_best_iteration", final_model.best_iteration)
        logger.info(f"Final Model Metrics: Accuracy={final_accuracy:.4f}, F1-Score={final_f1:.4f}, ROC AUC={final_roc_auc:.4f}")

        mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="final_xgboost_model", 
            registered_model_name=registered_model_name
        )
        logger.info(f"Final best model from run {mlflow.active_run().info.run_id} logged and registered as '{registered_model_name}'.")
        mlflow.set_tag("mlflow.source.name", f"train_flow.py (Best Model from Trial {best_hyperopt_run_id})")

    logger.info("Final model training and registration complete.")


@flow(name="Heart Disease Prediction Training Flow", log_prints=True)
def main_training_flow(
    data_file_path: str = "heart.csv",
    max_hyperopt_evals: int = 10,
    registered_model_name: str = "HeartDiseaseXGBoostModel"
):
    """
    Main Prefect flow for Heart Disease Prediction model training,
    including hyperparameter tuning and best model registration.
    """
    logger.info("Starting main training flow...")

    # 1. Load and Preprocess Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path=data_file_path)

    # 2. Define Hyperopt Search Space
    search_space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -4, -1),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 0),
        'reg_lambda': hp.loguniform('reg_lambda', -6, 0),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 2),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }
    
    # 3. Run Hyperopt Tuning (orchestrated by Prefect)
    # Hyperopt's fmin takes a callable. We use partial to pass static data to our objective task.
    # The objective task will then handle MLflow runs for each trial.
    objective_for_hyperopt = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    trials = Trials()
    
    logger.info(f"Initiating Hyperopt search with {max_hyperopt_evals} evaluations.")
    fmin(
        fn=objective_for_hyperopt,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_hyperopt_evals,
        trials=trials,
        rstate=None
    )
    logger.info("Hyperopt search completed.")

    # 4. Find the Best Run and Parameters (using a Prefect task)
    best_params, best_hyperopt_run_id = find_best_run_and_params(
        experiment_name="Heart Disease Prediction XGBoost Hyperopt"
    )

    # 5. Train and Register the Final Best Model (using a Prefect task)
    train_and_register_best_model(
        best_params=best_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        best_hyperopt_run_id=best_hyperopt_run_id,
        registered_model_name=registered_model_name
    )

    logger.info("Main training flow finished successfully!")

# Entry point for Prefect flow execution
if __name__ == "__main__":
    main_training_flow(data_file_path="heart.csv", max_hyperopt_evals=10)