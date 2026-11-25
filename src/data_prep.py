import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, PROCESSED_TRAIN_FILE, TRAIN_FILE

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
ADDRESS_MISMATCH_COLS = [
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
]


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Handle systemic placeholders discovered during EDA."""
    df = df.copy()

    # Flag and replace DAYS_EMPLOYED placeholder (365243 days ~ 1000 years)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Replace 'XNA' placeholders across categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("XNA", np.nan)

    # Replace rare 'Unknown' placeholder in family status (n=2 in train)
    if "NAME_FAMILY_STATUS" in df.columns:
        df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)

    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create banking ratios, behavioral stability features, and aggregations."""
    df = df.copy()

    # Row-level missingness count (thin-file risk signal)
    df["TOTAL_MISSING_CNT"] = df.isna().sum(axis=1).astype(np.int16)

    # External bureau score aggregations (EXT_SOURCE_MEAN IV = 0.608 in EDA)
    available_ext = [c for c in EXT_SOURCE_COLS if c in df.columns]
    if available_ext:
        df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[available_ext].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[available_ext].max(axis=1)
        df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1)
        df["EXT_SOURCE_PROD"] = df[available_ext].prod(axis=1, min_count=1)
        df["EXT_SOURCE_MISSING_CNT"] = df[available_ext].isna().sum(axis=1).astype(np.int8)

    # Credit & affordability ratios
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["GOODS_PRICE_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    # Demographic & behavioral stability ratios
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["PHONE_TO_AGE_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    df["ID_PUBLISH_TO_AGE_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]

    # Geo mismatch, social circle default rate, and document count
    available_mismatch = [c for c in ADDRESS_MISMATCH_COLS if c in df.columns]
    if available_mismatch:
        df["ADDRESS_MISMATCH_CNT"] = df[available_mismatch].sum(axis=1).astype(np.int8)

    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        df["SOCIAL_CIRCLE_DEF_RATIO"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"]
        )

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENTS_SUBMITTED_CNT"] = df[doc_cols].sum(axis=1).astype(np.int8)

    # Guard against division-by-zero infinities
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to pandas category dtype for LightGBM."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df


def preprocess_application(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for application data (train & inference)."""
    df = clean_anomalies(df)
    df = add_domain_features(df)
    df = cast_categoricals(df)
    return df


def main() -> None:
    print(f"Loading raw data from {TRAIN_FILE}...")
    raw_df = pd.read_csv(TRAIN_FILE)
    print(f"Raw shape: {raw_df.shape}")

    processed_df = preprocess_application(raw_df)
    print(f"Processed shape: {processed_df.shape}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(PROCESSED_TRAIN_FILE, index=False)
    print(f"Saved processed dataset to {PROCESSED_TRAIN_FILE}")


if __name__ == "__main__":
    main()