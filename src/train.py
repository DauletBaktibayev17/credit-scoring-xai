import argparse
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CATBOOST_BASELINE_PARAMS,
    ID_COL,
    LGBM_BASELINE_PARAMS,
    LOGREG_BINNING_CRITERIA,
    LOGREG_CORR_THRESHOLD,
    LOGREG_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    N_FOLDS,
    PROCESSED_TRAIN_FILE,
    PROCESSED_TRAIN_V2_FILE,
    RANDOM_STATE,
    TARGET_COL,
)
from src.metrics import evaluate_predictions


def drop_correlated_woe_features(
    X_woe: pd.DataFrame,
    iv_map: dict[str, float],
    threshold: float = 0.80,
) -> list[str]:
    """Drop collinear features on WoE matrix, keeping the higher-IV feature in each pair."""
    corr_matrix = X_woe.corr().abs()
    cols = list(X_woe.columns)
    to_drop: set[str] = set()

    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                iv_i = iv_map.get(cols[i], 0.0)
                iv_j = iv_map.get(cols[j], 0.0)
                to_drop.add(cols[j] if iv_i >= iv_j else cols[i])
                if cols[i] in to_drop:
                    break

    return [c for c in cols if c not in to_drop]


def train_logreg_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[dict, np.ndarray, int]:
    """Fit Optimal Binning + IV/correlation feature selection + Logistic Regression."""
    X_tr = X_train.copy()
    X_va = X_val.copy()

    # optbinning handles object/str categoricals cleanly (avoids pandas category issues)
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("object")
        X_va[col] = X_va[col].astype("object")

    binning = BinningProcess(
        variable_names=list(X_tr.columns),
        categorical_variables=cat_cols,
        selection_criteria=LOGREG_BINNING_CRITERIA,
        n_jobs=-1,
    )
    binning.fit(X_tr, y_train)

    summary = binning.summary()
    selected_by_iv = summary.loc[summary["selected"], "name"].tolist()
    iv_map = dict(zip(summary["name"], summary["iv"]))

    X_tr_woe = binning.transform(X_tr, metric="woe")[selected_by_iv]
    X_va_woe = binning.transform(X_va, metric="woe")[selected_by_iv]

    final_cols = drop_correlated_woe_features(X_tr_woe, iv_map, threshold=LOGREG_CORR_THRESHOLD)

    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_tr_woe[final_cols], y_train)
    val_preds = model.predict_proba(X_va_woe[final_cols])[:, 1]

    fold_artifact = {
        "binning": binning,
        "selected_cols": final_cols,
        "model": model,
    }
    return fold_artifact, val_preds, len(final_cols)


def train_lgbm_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[lgb.LGBMClassifier, np.ndarray, int]:
    """Fit LightGBM with native missing value and categorical support."""
    model = lgb.LGBMClassifier(**LGBM_BASELINE_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds, X_train.shape[1]


def train_catboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
) -> tuple[CatBoostClassifier, np.ndarray, int]:
    """Fit CatBoost with string-encoded categorical features."""
    X_tr = X_train.copy()
    X_va = X_val.copy()

    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("object").fillna("Missing").astype(str)
        X_va[col] = X_va[col].astype("object").fillna("Missing").astype(str)

    model = CatBoostClassifier(**CATBOOST_BASELINE_PARAMS)
    model.fit(
        X_tr,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_va, y_val),
        early_stopping_rounds=100,
        verbose=False,
    )
    val_preds = model.predict_proba(X_va)[:, 1]
    return model, val_preds, X_train.shape[1]


def run_training(model_name: str, feature_version: str = "v1_app_only") -> None:
    data_path = PROCESSED_TRAIN_V2_FILE if feature_version.startswith("v2") else PROCESSED_TRAIN_FILE
    print(f"Loading processed dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    feature_cols = [c for c in df.columns if c not in {ID_COL, TARGET_COL}]
    cat_cols = list(df[feature_cols].select_dtypes(include="category").columns)
    X = df[feature_cols]
    y = df[TARGET_COL].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(df), dtype=np.float64)
    fold_artifacts = []
    fold_aucs = []
    fold_n_features = []

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    start_time = time.time()
    with mlflow.start_run(run_name=f"{model_name}_{feature_version}"):
        mlflow.set_tags({
            "model_family": model_name,
            "feature_version": feature_version,
        })

        if model_name == "logreg":
            mlflow.log_params({
                **LOGREG_PARAMS,
                "iv_min": LOGREG_BINNING_CRITERIA["iv"]["min"],
                "corr_threshold": LOGREG_CORR_THRESHOLD,
            })
        elif model_name == "lgbm":
            mlflow.log_params(LGBM_BASELINE_PARAMS)
        elif model_name == "catboost":
            mlflow.log_params(CATBOOST_BASELINE_PARAMS)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            if model_name == "logreg":
                artifact, val_preds, n_feat = train_logreg_fold(X_train, y_train, X_val, cat_cols)
            elif model_name == "lgbm":
                artifact, val_preds, n_feat = train_lgbm_fold(X_train, y_train, X_val, y_val)
            elif model_name == "catboost":
                artifact, val_preds, n_feat = train_catboost_fold(X_train, y_train, X_val, y_val, cat_cols)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            oof_preds[val_idx] = val_preds
            fold_metrics = evaluate_predictions(y_val, val_preds)
            fold_aucs.append(fold_metrics["roc_auc"])
            fold_n_features.append(n_feat)
            fold_artifacts.append(artifact)

            print(
                f"[{model_name.upper()}] Fold {fold}/{N_FOLDS} | "
                f"AUC: {fold_metrics['roc_auc']:.5f} | Gini: {fold_metrics['gini']:.5f} | "
                f"KS: {fold_metrics['ks_stat']:.5f} | Features: {n_feat}"
            )

        elapsed_sec = round(time.time() - start_time, 1)
        oof_metrics = evaluate_predictions(y, oof_preds)
        oof_metrics["fold_auc_std"] = round(float(np.std(fold_aucs)), 5)
        oof_metrics["avg_n_features"] = round(float(np.mean(fold_n_features)), 1)
        oof_metrics["train_time_sec"] = elapsed_sec

        mlflow.log_metrics(oof_metrics)

        print(f"\n--- OOF Summary ({model_name.upper()} | {feature_version}) ---")
        for k, v in oof_metrics.items():
            print(f"{k:>16}: {v}")

        # Save model bundle & OOF predictions for blending / error analysis
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}_{feature_version}.joblib"
        oof_path = MODELS_DIR / f"oof_{model_name}_{feature_version}.parquet"

        joblib.dump(
            {
                "model_name": model_name,
                "feature_version": feature_version,
                "folds": fold_artifacts,
                "feature_cols": feature_cols,
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
    parser = argparse.ArgumentParser(description="Train credit scoring baseline models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logreg", "lgbm", "catboost"],
        help="Model family to train.",
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1_app_only",
        help="Feature set tag for MLflow tracking.",
    )
    args = parser.parse_args()
    run_training(model_name=args.model, feature_version=args.feature_version)