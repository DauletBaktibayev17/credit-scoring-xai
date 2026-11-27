import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ID_COL,
    LGBM_BASELINE_PARAMS,
    MODELS_DIR,
    N_FOLDS,
    PROCESSED_TRAIN_V2_FILE,
    RANDOM_STATE,
    TARGET_COL,
)


def evaluate_feature_subset(
    df: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    skf: StratifiedKFold,
) -> tuple[float, float]:
    """Quick 5-fold LightGBM CV (lr=0.05) to evaluate a feature subset."""
    params = {**LGBM_BASELINE_PARAMS, "learning_rate": 0.05}
    X = df[features]
    oof = np.zeros(len(df), dtype=np.float64)
    fold_aucs = []

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_X=X.iloc[val_idx],
            eval_y=y[val_idx],
            callbacks=[
                lgb.early_stopping(stopping_rounds=75, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = preds
        fold_aucs.append(roc_auc_score(y[val_idx], preds))

    return float(roc_auc_score(y, oof)), float(np.std(fold_aucs))


def run_feature_selection() -> None:
    model_path = MODELS_DIR / "lgbm_v2_relational.joblib"
    print(f"Loading v2 LightGBM bundle from {model_path}...")
    bundle = joblib.load(model_path)

    feature_cols = bundle["feature_cols"]
    models = bundle["folds"]

    mean_gain = np.mean(
        [m.booster_.feature_importance(importance_type="gain") for m in models],
        axis=0,
    )
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "gain": mean_gain})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["cum_gain_pct"] = fi_df["gain"].cumsum() / fi_df["gain"].sum()

    zero_cnt = int((fi_df["gain"] == 0).sum())
    nonzero_cols = fi_df.loc[fi_df["gain"] > 0, "feature"].tolist()

    print(f"\nTotal features: {len(feature_cols)} | Zero-gain features: {zero_cnt}")
    print("\nTop 20 Features by Mean 5-Fold Gain:")
    print(fi_df.head(20).to_string(index=False))

    # Save full feature importance table for reporting / README
    fi_path = MODELS_DIR / "feature_importance_v2.csv"
    fi_df.to_csv(fi_path, index=False)

    print(f"\nLoading {PROCESSED_TRAIN_V2_FILE} for subset CV sweep...")
    df = pd.read_parquet(PROCESSED_TRAIN_V2_FILE)
    y = df[TARGET_COL].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Evaluate candidate feature counts
    candidates = sorted({n for n in [len(nonzero_cols), 220, 180, 140, 100] if n <= len(nonzero_cols)}, reverse=True)
    results = []

    print("\nRunning 5-fold CV sweep across feature counts (lr=0.05):")
    for n_feat in candidates:
        t0 = time.time()
        subset = fi_df["feature"].iloc[:n_feat].tolist()
        oof_auc, fold_std = evaluate_feature_subset(df, subset, y, skf)
        elapsed = round(time.time() - t0, 1)
        results.append({
            "n_features": n_feat,
            "oof_roc_auc": round(oof_auc, 5),
            "oof_gini": round(2 * oof_auc - 1, 5),
            "fold_std": round(fold_std, 5),
            "time_sec": elapsed,
        })
        print(
            f"  Top {n_feat:>3} features | OOF AUC: {oof_auc:.5f} | "
            f"Gini: {2 * oof_auc - 1:.5f} | Std: {fold_std:.5f} | {elapsed}s"
        )

    res_df = pd.DataFrame(results)
    best_row = res_df.sort_values("oof_roc_auc", ascending=False).iloc[0]
    best_n = int(best_row["n_features"])
    selected_features = fi_df["feature"].iloc[:best_n].tolist()

    out_path = MODELS_DIR / "selected_features_v3.json"
    out_path.write_text(json.dumps(selected_features, indent=2))
    print(f"\nBest subset: Top {best_n} features (OOF AUC: {best_row['oof_roc_auc']:.5f})")
    print(f"Saved selected feature list to {out_path}")


if __name__ == "__main__":
    run_feature_selection()