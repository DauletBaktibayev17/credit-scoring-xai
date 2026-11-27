import argparse
import json
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ID_COL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    N_FOLDS,
    PROCESSED_TRAIN_V2_FILE,
    RANDOM_STATE,
    TARGET_COL,
)
from src.metrics import evaluate_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(X: pd.DataFrame, y: np.ndarray, skf: StratifiedKFold):
    """Build Optuna objective with 5-fold CV and fold-level median pruning."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 3000,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 250),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_X=X.iloc[val_idx],
                eval_y=y[val_idx],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=60, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[val_idx], preds))

            # Report running mean AUC to allow early pruning of weak trials
            trial.report(float(np.mean(fold_aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_aucs))

    return objective


def run_tuning(n_trials: int = 25) -> None:
    features_path = MODELS_DIR / "selected_features_v3.json"
    selected_features = json.loads(features_path.read_text())

    print(f"Loading {PROCESSED_TRAIN_V2_FILE} with {len(selected_features)} selected features...")
    df = pd.read_parquet(PROCESSED_TRAIN_V2_FILE)
    X = df[selected_features]
    y = df[TARGET_COL].values
    cat_cols = list(X.select_dtypes(include="category").columns)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\nStarting Optuna hyperparameter search ({n_trials} trials, 5-fold CV, lr=0.05)...")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def log_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        status = trial.state.name
        val_str = f"{trial.value:.5f}" if trial.value is not None else "PRUNED"
        best_str = f"{study.best_value:.5f}" if study.best_trial is not None else "N/A"
        print(
            f"  Trial {trial.number + 1:>2}/{n_trials} | {status:<8} | "
            f"AUC: {val_str} | Best: {best_str}"
        )

    study.optimize(
        make_objective(X, y, skf),
        n_trials=n_trials,
        callbacks=[log_trial_callback],
    )
    tune_sec = round(time.time() - t0, 1)

    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "n_estimators": 5000,
        "subsample_freq": 1,
        **study.best_params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    params_path = MODELS_DIR / "best_lgbm_params_v3.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"\nOptuna finished in {tune_sec}s | Best CV AUC (lr=0.05): {study.best_value:.5f}")
    print(f"Saved best params to {params_path}")

    # Train final 5-fold LightGBM with lower learning rate (0.02) and log to MLflow
    print("\nTraining final 5-fold LightGBM (v3_tuned, lr=0.02)...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    oof_preds = np.zeros(len(df), dtype=np.float64)
    fold_models = []
    fold_aucs = []
    train_start = time.time()

    with mlflow.start_run(run_name="lgbm_v3_tuned"):
        mlflow.set_tags({"model_family": "lgbm", "feature_version": "v3_tuned"})
        mlflow.log_params({**best_params, "optuna_trials": n_trials})

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            model = lgb.LGBMClassifier(**best_params)
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_X=X.iloc[val_idx],
                eval_y=y[val_idx],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=150, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            oof_preds[val_idx] = preds
            fold_metrics = evaluate_predictions(y[val_idx], preds)
            fold_aucs.append(fold_metrics["roc_auc"])
            fold_models.append(model)

            print(
                f"[LGBM v3_tuned] Fold {fold}/{N_FOLDS} | "
                f"AUC: {fold_metrics['roc_auc']:.5f} | Gini: {fold_metrics['gini']:.5f} | "
                f"KS: {fold_metrics['ks_stat']:.5f} | Best iter: {model.best_iteration_}"
            )

        elapsed_sec = round(time.time() - train_start, 1)
        oof_metrics = evaluate_predictions(y, oof_preds)
        oof_metrics["fold_auc_std"] = round(float(np.std(fold_aucs)), 5)
        oof_metrics["avg_n_features"] = float(len(selected_features))
        oof_metrics["train_time_sec"] = elapsed_sec

        mlflow.log_metrics(oof_metrics)

        print("\n--- OOF Summary (LGBM | v3_tuned) ---")
        for k, v in oof_metrics.items():
            print(f"{k:>16}: {v}")

        model_path = MODELS_DIR / "lgbm_v3_tuned.joblib"
        oof_path = MODELS_DIR / "oof_lgbm_v3_tuned.parquet"
        joblib.dump(
            {
                "model_name": "lgbm",
                "feature_version": "v3_tuned",
                "folds": fold_models,
                "feature_cols": selected_features,
                "cat_cols": cat_cols,
            },
            model_path,
        )
        pd.DataFrame({ID_COL: df[ID_COL], TARGET_COL: y, "oof_pred": oof_preds}).to_parquet(
            oof_path, index=False
        )
        print(f"Saved model to {model_path}")
        print(f"Saved OOF predictions to {oof_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune LightGBM with Optuna on v3 features.")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials.")
    args = parser.parse_args()
    run_tuning(n_trials=args.n_trials)