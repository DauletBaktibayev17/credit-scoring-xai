import argparse

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from src.config import (
    ID_COL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    TARGET_COL,
)
from src.metrics import evaluate_predictions


def optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 that maximize OOF ROC-AUC."""
    n_models = oof_matrix.shape[1]

    def loss(raw_weights: np.ndarray) -> float:
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / np.sum(exp_w)
        blend_pred = oof_matrix @ weights
        return -roc_auc_score(y_true, blend_pred)

    res = minimize(loss, x0=np.zeros(n_models), method="Nelder-Mead", options={"maxiter": 300})
    exp_w = np.exp(res.x - np.max(res.x))
    return exp_w / np.sum(exp_w)


def run_blend(model_tags: list[str], run_name: str) -> None:
    dfs = {}
    for tag in model_tags:
        path = MODELS_DIR / f"oof_{tag}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"OOF file not found: {path}")
        dfs[tag] = pd.read_parquet(path)

    base = dfs[model_tags[0]]
    y = base[TARGET_COL].values
    ids = base[ID_COL].values

    oof_df = pd.DataFrame({tag: dfs[tag]["oof_pred"].values for tag in model_tags})

    print("OOF prediction correlation matrix:")
    print(oof_df.corr().round(4))

    print("\nIndividual model OOF ROC-AUC:")
    for tag in model_tags:
        auc = roc_auc_score(y, oof_df[tag].values)
        print(f"  {tag:<28}: {auc:.5f}")

    weights = optimize_weights(oof_df.values, y)
    blend_pred = oof_df.values @ weights
    metrics = evaluate_predictions(y, blend_pred)

    print("\nOptimal Blend Weights:")
    weight_dict = {}
    for tag, w in zip(model_tags, weights):
        weight_dict[f"weight_{tag}"] = round(float(w), 4)
        print(f"  {tag:<28}: {w:.4f}")

    print(f"\n--- OOF Summary (BLEND | {run_name}) ---")
    for k, v in metrics.items():
        print(f"{k:>16}: {v}")

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_family": "blend", "feature_version": run_name})
        mlflow.log_params(weight_dict)
        mlflow.log_metrics(metrics)

    out_path = MODELS_DIR / f"oof_{run_name}.parquet"
    pd.DataFrame({ID_COL: ids, TARGET_COL: y, "oof_pred": blend_pred}).to_parquet(
        out_path, index=False
    )
    print(f"Saved blended OOF predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize probability blend of OOF predictions.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model tags matching models/oof_<tag>.parquet (e.g. lgbm_v2_relational catboost_v2_relational).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="blend_v2_relational",
        help="MLflow run name and output file tag.",
    )
    args = parser.parse_args()
    run_blend(model_tags=args.models, run_name=args.run_name)