from pathlib import Path
import gc

import numpy as np
import pandas as pd

PREV_DAYS_ANOM_COLS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def _flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Flatten MultiIndex columns from groupby.agg and apply a prefix."""
    df.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in df.columns]
    return df.reset_index()


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN and downcast float64 to float32 to save RAM."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def build_bureau_features(data_dir: Path) -> pd.DataFrame:
    """Aggregate bureau.csv and bureau_balance.csv to SK_ID_CURR level."""
    print("  [1/5] Aggregating bureau + bureau_balance...")
    bb = pd.read_csv(
        data_dir / "bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    )
    bb["IS_DPD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int8)
    bb["IS_DPD_12M"] = ((bb["MONTHS_BALANCE"] >= -12) & (bb["IS_DPD"] == 1)).astype(np.int8)
    bb["IS_CLOSED_STATUS"] = (bb["STATUS"] == "C").astype(np.int8)

    bb_agg = _flatten_columns(
        bb.groupby("SK_ID_BUREAU").agg(
            {
                "MONTHS_BALANCE": ["min", "max", "size"],
                "IS_DPD": ["mean", "max", "sum"],
                "IS_DPD_12M": ["max", "sum"],
                "IS_CLOSED_STATUS": ["mean"],
            }
        ),
        prefix="BB",
    )
    del bb
    gc.collect()

    bureau = pd.read_csv(data_dir / "bureau.csv")
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb_agg
    gc.collect()

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    bureau["IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(np.int8)
    bureau["DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    bureau["OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
    bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["ENDDATE_DIFF"] = bureau["DAYS_ENDDATE_FACT"] - bureau["DAYS_CREDIT_ENDDATE"]

    # Overall bureau aggregations
    agg_all = _flatten_columns(
        bureau.groupby("SK_ID_CURR").agg(
            {
                "SK_ID_BUREAU": ["count"],
                "IS_ACTIVE": ["sum", "mean"],
                "IS_CLOSED": ["sum", "mean"],
                "DAYS_CREDIT": ["min", "max", "mean"],
                "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
                "DAYS_CREDIT_UPDATE": ["max", "mean"],
                "CREDIT_DAY_OVERDUE": ["max", "mean"],
                "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
                "AMT_CREDIT_SUM": ["sum", "mean", "max"],
                "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
                "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
                "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],
                "CNT_CREDIT_PROLONG": ["sum", "max"],
                "DEBT_RATIO": ["mean", "max"],
                "OVERDUE_RATIO": ["mean", "max"],
                "CREDIT_DURATION": ["mean", "max"],
                "ENDDATE_DIFF": ["mean"],
                "BB_IS_DPD_MAX": ["max", "mean"],
                "BB_IS_DPD_12M_MAX": ["max"],
                "BB_IS_DPD_SUM": ["sum"],
                "BB_MONTHS_BALANCE_SIZE": ["mean", "sum"],
            }
        ),
        prefix="BUREAU",
    )

    # Total client-level debt-to-credit ratio
    agg_all["BUREAU_TOTAL_DEBT_TO_CREDIT"] = (
        agg_all["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"]
        / agg_all["BUREAU_AMT_CREDIT_SUM_SUM"].replace(0, np.nan)
    )

    # Active loans only aggregations
    active = bureau[bureau["IS_ACTIVE"] == 1]
    agg_active = _flatten_columns(
        active.groupby("SK_ID_CURR").agg(
            {
                "DAYS_CREDIT": ["min", "max", "mean"],
                "DAYS_CREDIT_ENDDATE": ["max", "mean"],
                "AMT_CREDIT_SUM": ["sum", "mean"],
                "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
                "DEBT_RATIO": ["mean", "max"],
            }
        ),
        prefix="BUREAU_ACTIVE",
    )

    res = agg_all.merge(agg_active, on="SK_ID_CURR", how="left")
    del bureau, active, agg_all, agg_active
    gc.collect()
    return _downcast_numeric(res)


def build_prev_app_features(data_dir: Path) -> pd.DataFrame:
    """Aggregate previous_application.csv with 365243 placeholder cleaning."""
    print("  [2/5] Aggregating previous_application...")
    prev = pd.read_csv(data_dir / "previous_application.csv")

    for col in PREV_DAYS_ANOM_COLS:
        if col in prev.columns:
            prev[col] = prev[col].replace(365243, np.nan)

    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    prev["IS_CANCELED"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)
    prev["APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["CREDIT_TERM"] = prev["AMT_ANNUITY"] / prev["AMT_CREDIT"].replace(0, np.nan)

    agg_all = _flatten_columns(
        prev.groupby("SK_ID_CURR").agg(
            {
                "SK_ID_PREV": ["count"],
                "IS_APPROVED": ["sum", "mean"],
                "IS_REFUSED": ["sum", "mean"],
                "IS_CANCELED": ["sum", "mean"],
                "AMT_ANNUITY": ["mean", "max", "min"],
                "AMT_APPLICATION": ["mean", "max", "sum"],
                "AMT_CREDIT": ["mean", "max", "sum"],
                "AMT_DOWN_PAYMENT": ["mean", "max"],
                "RATE_DOWN_PAYMENT": ["mean", "max"],
                "APP_CREDIT_RATIO": ["mean", "max", "min"],
                "APP_CREDIT_DIFF": ["mean", "min", "max"],
                "CREDIT_TERM": ["mean", "max"],
                "DAYS_DECISION": ["min", "max", "mean"],
                "CNT_PAYMENT": ["mean", "max", "sum"],
                "DAYS_LAST_DUE_1ST_VERSION": ["max", "mean"],
                "DAYS_TERMINATION": ["max", "mean"],
            }
        ),
        prefix="PREV",
    )

    # Approved applications only
    approved = prev[prev["IS_APPROVED"] == 1]
    agg_approved = _flatten_columns(
        approved.groupby("SK_ID_CURR").agg(
            {
                "AMT_CREDIT": ["mean", "max", "sum"],
                "AMT_ANNUITY": ["mean", "max"],
                "CNT_PAYMENT": ["mean", "max"],
                "DAYS_DECISION": ["max", "mean"],
            }
        ),
        prefix="PREV_APPROVED",
    )

    # Refused applications only
    refused = prev[prev["IS_REFUSED"] == 1]
    agg_refused = _flatten_columns(
        refused.groupby("SK_ID_CURR").agg(
            {
                "AMT_APPLICATION": ["mean", "max"],
                "DAYS_DECISION": ["max", "mean"],
            }
        ),
        prefix="PREV_REFUSED",
    )

    res = agg_all.merge(agg_approved, on="SK_ID_CURR", how="left").merge(
        agg_refused, on="SK_ID_CURR", how="left"
    )
    del prev, approved, refused, agg_all, agg_approved, agg_refused
    gc.collect()
    return _downcast_numeric(res)


def build_installments_features(data_dir: Path) -> pd.DataFrame:
    """Aggregate installments_payments.csv (overall + last 12 months)."""
    print("  [3/5] Aggregating installments_payments...")
    inst = pd.read_csv(
        data_dir / "installments_payments.csv",
        usecols=[
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )
    inst["DAYS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["DAYS_EARLY"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(np.int8)
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    inst["IS_UNDERPAID"] = (inst["PAYMENT_DIFF"] > 0.01).astype(np.int8)

    agg_all = _flatten_columns(
        inst.groupby("SK_ID_CURR").agg(
            {
                "DAYS_LATE": ["max", "mean", "sum"],
                "DAYS_EARLY": ["mean", "max"],
                "IS_LATE": ["mean", "sum"],
                "PAYMENT_DIFF": ["mean", "max", "sum"],
                "PAYMENT_RATIO": ["mean", "min"],
                "IS_UNDERPAID": ["mean", "sum"],
                "AMT_PAYMENT": ["sum", "mean", "max"],
                "DAYS_INSTALMENT": ["min", "max", "count"],
            }
        ),
        prefix="INST",
    )

    # Recent 12 months payment discipline
    recent = inst[inst["DAYS_INSTALMENT"] >= -365]
    agg_recent = _flatten_columns(
        recent.groupby("SK_ID_CURR").agg(
            {
                "DAYS_LATE": ["max", "mean"],
                "IS_LATE": ["mean", "sum"],
                "IS_UNDERPAID": ["mean", "sum"],
                "PAYMENT_RATIO": ["mean", "min"],
            }
        ),
        prefix="INST_12M",
    )

    res = agg_all.merge(agg_recent, on="SK_ID_CURR", how="left")
    del inst, recent, agg_all, agg_recent
    gc.collect()
    return _downcast_numeric(res)


def build_pos_cash_features(data_dir: Path) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv."""
    print("  [4/5] Aggregating POS_CASH_balance...")
    pos = pd.read_csv(
        data_dir / "POS_CASH_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "NAME_CONTRACT_STATUS",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
    )
    pos["IS_DPD"] = (pos["SK_DPD"] > 0).astype(np.int8)
    pos["IS_DPD_DEF"] = (pos["SK_DPD_DEF"] > 0).astype(np.int8)
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(np.int8)
    pos["REMAINING_RATIO"] = pos["CNT_INSTALMENT_FUTURE"] / pos["CNT_INSTALMENT"].replace(0, np.nan)

    res = _flatten_columns(
        pos.groupby("SK_ID_CURR").agg(
            {
                "MONTHS_BALANCE": ["max", "min", "size"],
                "SK_DPD": ["max", "mean"],
                "SK_DPD_DEF": ["max", "mean"],
                "IS_DPD": ["mean", "sum"],
                "IS_DPD_DEF": ["mean", "sum"],
                "IS_COMPLETED": ["mean", "sum"],
                "CNT_INSTALMENT_FUTURE": ["mean", "min"],
                "REMAINING_RATIO": ["mean"],
            }
        ),
        prefix="POS",
    )
    del pos
    gc.collect()
    return _downcast_numeric(res)


def build_credit_card_features(data_dir: Path) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv (overall + last 12 months)."""
    print("  [5/5] Aggregating credit_card_balance...")
    cc = pd.read_csv(
        data_dir / "credit_card_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_CURRENT",
            "CNT_DRAWINGS_ATM_CURRENT",
            "CNT_DRAWINGS_CURRENT",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
    )
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["IS_OVERLIMIT"] = (cc["UTILIZATION"] > 1.0).astype(np.int8)
    cc["ATM_SHARE"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / cc["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    cc["IS_DPD"] = (cc["SK_DPD"] > 0).astype(np.int8)

    agg_all = _flatten_columns(
        cc.groupby("SK_ID_CURR").agg(
            {
                "MONTHS_BALANCE": ["min", "max", "size"],
                "AMT_BALANCE": ["mean", "max"],
                "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
                "UTILIZATION": ["mean", "max", "min"],
                "IS_OVERLIMIT": ["mean", "sum"],
                "AMT_DRAWINGS_ATM_CURRENT": ["mean", "sum"],
                "CNT_DRAWINGS_ATM_CURRENT": ["mean", "max"],
                "CNT_DRAWINGS_CURRENT": ["mean", "max"],
                "ATM_SHARE": ["mean"],
                "SK_DPD": ["max", "mean"],
                "SK_DPD_DEF": ["max", "mean"],
                "IS_DPD": ["mean", "sum"],
            }
        ),
        prefix="CC",
    )

    # Recent 12 months utilization & balance
    recent = cc[cc["MONTHS_BALANCE"] >= -12]
    agg_recent = _flatten_columns(
        recent.groupby("SK_ID_CURR").agg(
            {
                "UTILIZATION": ["mean", "max"],
                "AMT_BALANCE": ["mean", "max"],
                "IS_OVERLIMIT": ["mean", "max"],
                "CNT_DRAWINGS_ATM_CURRENT": ["mean"],
            }
        ),
        prefix="CC_12M",
    )

    res = agg_all.merge(agg_recent, on="SK_ID_CURR", how="left")
    del cc, recent, agg_all, agg_recent
    gc.collect()
    return _downcast_numeric(res)


def merge_all_relational_features(app_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Build and left-join all relational table features onto application dataframe."""
    builders = [
        build_bureau_features,
        build_prev_app_features,
        build_installments_features,
        build_pos_cash_features,
        build_credit_card_features,
    ]
    df = app_df.copy()
    for builder in builders:
        rel_df = builder(data_dir)
        df = df.merge(rel_df, on="SK_ID_CURR", how="left")
        del rel_df
        gc.collect()
    return df