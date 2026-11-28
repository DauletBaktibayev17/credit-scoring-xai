from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COL, MODELS_DIR, PROCESSED_TRAIN_V2_FILE, PROJECT_ROOT, TARGET_COL
from src.data_prep import preprocess_application
from src.features_relational import merge_all_relational_features

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_TEST_V2_FILE = PROJECT_ROOT / "data" / "processed" / "test_features_v2.parquet"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Optimal OOF weights from src/blend.py (blend_v3_final)
W_LGBM = 0.6325
W_CATBOOST = 0.3675


def build_or_load_test_features() -> pd.DataFrame:
    if PROCESSED_TEST_V2_FILE.exists():
        print(f"Loading cached test features from {PROCESSED_TEST_V2_FILE}...")
        return pd.read_parquet(PROCESSED_TEST_V2_FILE)

    test_raw_path = RAW_DATA_DIR / "application_test.csv"
    print(f"Loading raw test data from {test_raw_path}...")
    df = pd.read_csv(test_raw_path)
    print(f"Raw test shape: {df.shape}")

    df = preprocess_application(df)
    print("Building and merging relational features (v2) for test set...")
    df = merge_all_relational_features(df, RAW_DATA_DIR)
    print(f"Processed test shape (v2): {df.shape}")

    PROCESSED_TEST_V2_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_TEST_V2_FILE, index=False)
    print(f"Saved processed test dataset to {PROCESSED_TEST_V2_FILE}")
    return df


def align_lgbm_categories(
    X_test: pd.DataFrame,
    cat_cols: list[str],
    train_cat_schema: dict[str, pd.Index],
) -> pd.DataFrame:
    """Align test pandas categorical dtypes to exact training categories for LightGBM."""
    X = X_test.copy()
    for col in cat_cols:
        if col in train_cat_schema:
            X[col] = pd.Categorical(X[col].astype("object"), categories=train_cat_schema[col])
    return X


def prepare_catboost_frame(X_test: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Apply exact string encoding used during CatBoost training."""
    X = X_test.copy()
    for col in cat_cols:
        X[col] = X[col].astype("object").fillna("Missing").astype(str)
    return X


def predict_5fold(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(X), dtype=np.float64)
    for model in models:
        preds += model.predict_proba(X)[:, 1] / len(models)
    return preds


def main() -> None:
    test_df = build_or_load_test_features()

    # Load only categorical columns from train to extract exact category levels
    lgbm_bundle = joblib.load(MODELS_DIR / "lgbm_v3_tuned.joblib")
    cb_bundle = joblib.load(MODELS_DIR / "catboost_v2_relational.joblib")

    lgbm_feats: list[str] = lgbm_bundle["feature_cols"]
    lgbm_cats: list[str] = lgbm_bundle["cat_cols"]
    cb_feats: list[str] = cb_bundle["feature_cols"]
    cb_cats: list[str] = cb_bundle["cat_cols"]

    train_cats_df = pd.read_parquet(PROCESSED_TRAIN_V2_FILE, columns=lgbm_cats)
    train_cat_schema = {
        col: train_cats_df[col].cat.categories
        for col in lgbm_cats
        if isinstance(train_cats_df[col].dtype, pd.CategoricalDtype)
    }

    # Ensure any missing columns (edge case if a rare category/agg is absent) exist as NaN
    for col in set(lgbm_feats) | set(cb_feats):
        if col not in test_df.columns:
            test_df[col] = np.nan

    print(f"\nPredicting with 5-fold LightGBM v3_tuned ({len(lgbm_feats)} features)...")
    X_lgbm = align_lgbm_categories(test_df[lgbm_feats], lgbm_cats, train_cat_schema)
    lgbm_preds = predict_5fold(lgbm_bundle["folds"], X_lgbm)

    print(f"Predicting with 5-fold CatBoost v2_relational ({len(cb_feats)} features)...")
    X_cb = prepare_catboost_frame(test_df[cb_feats], cb_cats)
    cb_preds = predict_5fold(cb_bundle["folds"], X_cb)

    blend_preds = W_LGBM * lgbm_preds + W_CATBOOST * cb_preds

    # Sanity checks
    corr = float(np.corrcoef(lgbm_preds, cb_preds)[0, 1])
    print("\n--- Test Prediction Diagnostics ---")
    print(f"  Rows               : {len(blend_preds):,} (expected 48,744)")
    print(f"  NaN count          : {int(np.isnan(blend_preds).sum())}")
    print(f"  LGBM-CatBoost corr : {corr:.4f} (OOF was 0.9634)")
    print(f"  LGBM mean PD       : {lgbm_preds.mean():.5f}")
    print(f"  CatBoost mean PD   : {cb_preds.mean():.5f}")
    print(f"  Blend mean PD      : {blend_preds.mean():.5f} | min: {blend_preds.min():.5f} | max: {blend_preds.max():.5f}")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "submission_blend_v3.csv"
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].astype(int), TARGET_COL: blend_preds})
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved Kaggle submission to {sub_path}")


if __name__ == "__main__":
    main()