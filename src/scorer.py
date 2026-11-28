from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR

# Decision thresholds calibrated on OOF deciles (base default rate ~8.07%)
APPROVE_THRESHOLD = 0.08
DECLINE_THRESHOLD = 0.20


def map_feature_to_reason_group(feature: str) -> str:
    """Group technical features into business-friendly Adverse Action Reason Codes."""
    if feature.startswith("EXT_SOURCE"):
        return "External Credit Bureau Score"
    if feature.startswith("CC_"):
        return "Credit Card Utilization & Cash Advances"
    if feature.startswith("BUREAU_"):
        return "External Bureau Debt & History"
    if feature.startswith("INST_"):
        return "Installment Payment Discipline"
    if feature.startswith("PREV_"):
        return "Previous Bank Applications & Refusals"
    if feature.startswith("POS_"):
        return "POS / Cash Loan History"
    if feature in {
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "AMT_INCOME_TOTAL",
        "CREDIT_TERM",
        "CREDIT_GOODS_RATIO",
        "GOODS_PRICE_DIFF",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_INCOME_RATIO",
    }:
        return "Loan Affordability & Terms"
    if feature in {
        "AGE_YEARS",
        "DAYS_BIRTH",
        "EMPLOYED_YEARS",
        "DAYS_EMPLOYED",
        "EMPLOYED_TO_AGE_RATIO",
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "CODE_GENDER",
        "OWN_CAR_AGE",
    }:
        return "Applicant Profile & Employment"
    return feature.replace("_", " ").title()


@dataclass
class ScoringResult:
    pd_score: float
    decision: str
    risk_band: str
    base_value: float
    reason_codes: pd.DataFrame
    feature_shap: pd.DataFrame


class CreditScorer:
    """Production inference & SHAP explanation wrapper for 5-fold LightGBM."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (MODELS_DIR / "lgbm_v3_tuned.joblib")
        bundle = joblib.load(self.model_path)
        self.folds = bundle["folds"]
        self.feature_cols: list[str] = bundle["feature_cols"]
        self.cat_cols: list[str] = bundle["cat_cols"]
        # Use fold-0 booster for fast local TreeSHAP explanations
        self.explainer = shap.TreeExplainer(self.folds[0])

    @staticmethod
    def _classify_decision(pd_score: float) -> tuple[str, str]:
        if pd_score < APPROVE_THRESHOLD:
            return "APPROVE", "Low Risk"
        if pd_score < DECLINE_THRESHOLD:
            return "MANUAL_REVIEW", "Medium Risk"
        return "DECLINE", "High Risk"

    def score_applicant(self, row: pd.DataFrame, top_k: int = 5) -> ScoringResult:
        """Compute 5-fold ensemble PD and grouped + raw SHAP explanations for 1 applicant."""
        X_row = row[self.feature_cols].copy()

        # Average predicted probability across all 5 CV folds
        fold_preds = [float(m.predict_proba(X_row)[0, 1]) for m in self.folds]
        pd_score = float(np.mean(fold_preds))
        decision, risk_band = self._classify_decision(pd_score)

        # Compute local TreeSHAP
        shap_exp = self.explainer(X_row)
        if shap_exp.values.ndim == 3:
            sv = shap_exp.values[0, :, 1]
            base_val = float(shap_exp.base_values[0, 1])
        else:
            sv = shap_exp.values[0]
            base_val = float(np.ravel(shap_exp.base_values)[0])

        feature_df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "value": [X_row.iloc[0][c] for c in self.feature_cols],
                "shap_value": sv,
                "abs_shap": np.abs(sv),
                "reason_group": [map_feature_to_reason_group(c) for c in self.feature_cols],
            }
        ).sort_values("abs_shap", ascending=False)

        # Aggregate SHAP impacts into business reason groups
        grouped_df = (
            feature_df.groupby("reason_group", as_index=False)["shap_value"]
            .sum()
            .assign(abs_shap=lambda d: d["shap_value"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(top_k)
        )
        grouped_df["direction"] = np.where(
            grouped_df["shap_value"] > 0, "Increases Risk (+)", "Decreases Risk (-)"
        )

        return ScoringResult(
            pd_score=pd_score,
            decision=decision,
            risk_band=risk_band,
            base_value=base_val,
            reason_codes=grouped_df[["reason_group", "shap_value", "direction"]].reset_index(drop=True),
            feature_shap=feature_df.reset_index(drop=True),
        )