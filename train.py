import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
# mlflow.set_tracking_uri("http://localhost:5000") # Default

# Set experiment name
mlflow.set_experiment("Heart Disease Prediction XGBoost Hyperopt")
logger.info("MLflow experiment set to 'Heart Disease Prediction XGBoost Hyperopt'.")

def load_and_preprocess_data(file_path='heart.csv', random_state=42):
    """Loads and preprocesses the heart disease dataset."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Please ensure the file is in the same directory.")
        raise

    # Preprocessing: Convert categorical features to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    logger.info("Categorical features one-hot encoded.")

    # Define features (X) and target (y)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    return X_train, X_test, y_train, y_test

def objective(params, X_train, y_train, X_test, y_test):
    """
    Objective function for Hyperopt. Trains an XGBoost model and logs metrics to MLflow.
    Hyperopt minimizes the 'loss' value. For classification, we use (1 - ROC AUC).
    This function *does not* log the model artifact.
    """
    with mlflow.start_run():
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("tuning_method", "hyperopt_trial") # Tag to identify these runs later
        
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
                num_boost_round=100, # Max boosting rounds, early stopping will prevent overfitting
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

def run_hyperopt_tuning():
    """Main function to run hyperparameter tuning with Hyperopt, then train and log the best model."""
    logger.info("Starting Hyperopt tuning process...")

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Define a smaller search space for Hyperopt
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

    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    trials = Trials()
    max_evals = 10 
    logger.info(f"Running Hyperopt with max_evals={max_evals}")

    best_result_hyperopt = fmin(
        fn=objective_with_data,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=None # Use global random state for reproducibility of hyperopt search
    )
    logger.info("Hyperopt tuning finished.")
    logger.info(f"Best parameters found by Hyperopt: {best_result_hyperopt}")

    # Find the best run based on metrics from the MLflow Tracking Server
    client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow.get_experiment_by_name("Heart Disease Prediction XGBoost Hyperopt").experiment_id
    
    logger.info("Searching for the best hyperopt trial within the experiment.")

    # We filter by the tag 'tuning_method' to ensure we only pick hyperopt trials.
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.roc_auc DESC"], # Order by ROC AUC to find the best
        max_results=1,
        filter_string="attributes.status = 'FINISHED' AND tags.\"tuning_method\" = 'hyperopt_trial'"
    )
    
    if not runs:
        logger.error(f"No successful hyperopt trials found within experiment {experiment_id}. Ensure trials are finishing with status 'FINISHED' and tag 'tuning_method' as 'hyperopt_trial'.")
        return

    best_hyperopt_run = runs[0]
    # Extract best parameters from the run data (more reliable than best_result_hyperopt from fmin)
    best_params_from_run = {k: v for k, v in best_hyperopt_run.data.params.items()}
    
    # Ensure max_depth is int for final training
    if 'max_depth' in best_params_from_run:
        best_params_from_run['max_depth'] = int(best_params_from_run['max_depth'])
    
    logger.info(f"Best Hyperopt trial (based on ROC AUC): run_id={best_hyperopt_run.info.run_id}, ROC AUC={best_hyperopt_run.data.metrics['roc_auc']:.4f}")
    logger.info(f"Best parameters for final model training: {best_params_from_run}")

    # --- Train and log the FINAL BEST MODEL in a NEW, DEDICATED MLflow run ---
    logger.info("Training and logging the final best model.")

    with mlflow.start_run(run_name="Final_Best_XGBoost_Model") as final_run:
        mlflow.set_tag("mlops_step", "final_model_training")
        mlflow.log_params(best_params_from_run)

        # Convert data to DMatrix for final training
        dtrain_final = xgb.DMatrix(X_train, label=y_train)
        dtest_final = xgb.DMatrix(X_test, label=y_test)

        final_model = xgb.train(
            params=best_params_from_run,
            dtrain=dtrain_final,
            num_boost_round=100,
            evals=[(dtrain_final, 'train'), (dtest_final, 'eval')],
            verbose_eval=False,
            early_stopping_rounds=10
        )
        logger.info("Final best XGBoost model trained.")

        # Make predictions and evaluate for the final model run
        y_pred_proba_final = final_model.predict(dtest_final)
        y_pred_final = (y_pred_proba_final > 0.5).astype(int)

        final_accuracy = accuracy_score(y_test, y_pred_final)
        final_precision = precision_score(y_test, y_pred_final, zero_division=0)
        final_recall = recall_score(y_test, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_test, y_pred_final, zero_division=0)
        final_roc_auc = roc_auc_score(y_test, y_pred_proba_final)

        mlflow.log_metric("accuracy", final_accuracy)
        mlflow.log_metric("precision", final_precision)
        mlflow.log_metric("recall", final_recall)
        mlflow.log_metric("f1_score", final_f1)
        mlflow.log_metric("roc_auc", final_roc_auc)
        mlflow.log_metric("best_iteration", final_model.best_iteration)
        logger.info(f"Final Model Metrics: Accuracy={final_accuracy:.4f}, F1-Score={final_f1:.4f}, ROC AUC={final_roc_auc:.4f}")

        # Log and REGISTER the FINAL BEST MODEL
        mlflow.xgboost.log_model(
            xgb_model=final_model,
            artifact_path="final_xgboost_model", # Dedicated artifact path for the final model
            registered_model_name="HeartDiseaseXGBoostModel" # Register this one
        )
        logger.info(f"Final best model from run {final_run.info.run_id} logged and registered as 'HeartDiseaseXGBoostModel'.")
        mlflow.set_tag("mlflow.source.name", f"train.py (Best Model from Trial {best_hyperopt_run.info.run_id})")

    logger.info("Process complete. Check MLflow UI for results.")

if __name__ == "__main__":
    run_hyperopt_tuning()