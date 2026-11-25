from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Files
TRAIN_FILE = DATA_RAW_DIR / "application_train.csv"
PROCESSED_TRAIN_FILE = DATA_PROCESSED_DIR / "train_features.parquet"

# MLflow tracking
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "credit-scoring-baseline"

# Modeling constants
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_STATE = 42
N_FOLDS = 5

# Logistic Regression (with Optimal Binning) settings
LOGREG_BINNING_CRITERIA = {
    "iv": {"min": 0.02, "max": 1.0},
}
LOGREG_CORR_THRESHOLD = 0.80
LOGREG_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

# LightGBM baseline hyperparameters
LGBM_BASELINE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# CatBoost baseline hyperparameters
CATBOOST_BASELINE_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "iterations": 3000,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": -1,
}