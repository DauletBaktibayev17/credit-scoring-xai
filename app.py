from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, PROCESSED_TRAIN_V2_FILE, PROJECT_ROOT
from src.scorer import APPROVE_THRESHOLD, DECLINE_THRESHOLD, CreditScorer

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Credit Scoring XAI Dashboard",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_scorer() -> CreditScorer:
    return CreditScorer()


@st.cache_data
def load_demo_portfolio(sample_size: int = 500) -> pd.DataFrame:
    """Load a stratified sample of applicants across risk levels (plus key showcase IDs)."""
    df = pd.read_parquet(PROCESSED_TRAIN_V2_FILE)
    oof = pd.read_parquet(MODELS_DIR / "oof_lgbm_v3_tuned.parquet")[["SK_ID_CURR", "oof_pred"]]
    df = df.merge(oof, on="SK_ID_CURR", how="inner")

    showcase_ids = [395476, 171402]
    showcase_df = df[df["SK_ID_CURR"].isin(showcase_ids)]
    rest_df = df[~df["SK_ID_CURR"].isin(showcase_ids)].sample(n=sample_size, random_state=42)

    portfolio = pd.concat([showcase_df, rest_df], ignore_index=True)
    return portfolio.sort_values("oof_pred", ascending=False).reset_index(drop=True)


def format_value(val) -> str:
    """Format mixed numeric/categorical values as strings for PyArrow compatibility."""
    if pd.isna(val):
        return "NaN"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        return f"{val:,.4f}" if abs(val) < 10 else f"{val:,.2f}"
    return str(val)


def format_reason_table(reason_df: pd.DataFrame) -> pd.DataFrame:
    out = reason_df.copy()
    out["direction"] = np.where(out["shap_value"] > 0, "🔴 ↑ Risk", "🟢 ↓ Risk")
    out.columns = ["Reason Category", "Net SHAP", "Effect"]
    return out


def plot_shap_bar(feature_shap: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    top_df = feature_shap.head(top_n).iloc[::-1]
    colors = ["#E05A5A" if v > 0 else "#4C8BF5" for v in top_df["shap_value"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor("#161A23")
    ax.set_facecolor("#161A23")

    ax.barh(top_df["feature"], top_df["shap_value"], color=colors, height=0.68)
    ax.axvline(0, color="#8A94A6", linewidth=0.9)
    ax.set_xlabel("SHAP Value (Impact on Log-Odds of Default)", color="#E2E8F0", fontsize=10)
    ax.set_title(f"Top {top_n} Individual Feature Drivers", color="#F8FAFC", fontsize=11, pad=10)
    ax.tick_params(colors="#CBD5E1", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#334155")
    ax.grid(axis="x", linestyle="--", alpha=0.2, color="#94A3B8")
    plt.tight_layout()
    return fig


def render_decision_header(
    pd_score: float,
    decision: str,
    risk_band: str,
    actual_target: int | None = None,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted PD (5-Fold)", f"{pd_score:.2%}")
    c2.metric("Risk Band", risk_band)

    if decision == "APPROVE":
        c3.success(f"Decision: {decision}")
    elif decision == "MANUAL_REVIEW":
        c3.warning(f"Decision: {decision}")
    else:
        c3.error(f"Decision: {decision}")

    if actual_target is not None:
        label = "Default (1)" if actual_target == 1 else "Repaid (0)"
        c4.metric("Historical Outcome", label)
    else:
        c4.metric("Policy Cutoffs", f"<{APPROVE_THRESHOLD:.0%} / >{DECLINE_THRESHOLD:.0%}")


def main() -> None:
    st.title("🏦 Credit Scoring & Explainable AI (XAI) Dashboard")
    st.caption(
        "5-Fold Tuned LightGBM (220 relational & application features | OOF ROC-AUC: 0.7957 | Blend AUC: 0.7967)"
    )

    scorer = load_scorer()
    portfolio = load_demo_portfolio()

    # Sidebar applicant selector
    st.sidebar.header("Applicant Selection")
    risk_filter = st.sidebar.radio(
        "Filter by Risk Segment",
        ["All", "High Risk (>20%)", "Medium Risk (8–20%)", "Low Risk (<8%)"],
    )

    filtered = portfolio.copy()
    if risk_filter == "High Risk (>20%)":
        filtered = filtered[filtered["oof_pred"] >= DECLINE_THRESHOLD]
    elif risk_filter == "Medium Risk (8–20%)":
        filtered = filtered[
            (filtered["oof_pred"] >= APPROVE_THRESHOLD) & (filtered["oof_pred"] < DECLINE_THRESHOLD)
        ]
    elif risk_filter == "Low Risk (<8%)":
        filtered = filtered[filtered["oof_pred"] < APPROVE_THRESHOLD]

    options = [f"ID {int(r.SK_ID_CURR)} | OOF PD: {r.oof_pred:.1%}" for r in filtered.itertuples()]
    selected_label = st.sidebar.selectbox("Choose Applicant", options)
    selected_id = int(selected_label.split()[1])
    applicant_row = portfolio[portfolio["SK_ID_CURR"] == selected_id].iloc[[0]].copy()
    actual_target = int(applicant_row["TARGET"].iloc[0])

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Applicant Review & Reason Codes",
            "🎛️ What-If Risk Simulator",
            "📊 Model & Portfolio Validation",
        ]
    )

    with tab1:
        result = scorer.score_applicant(applicant_row, top_k=5)
        render_decision_header(result.pd_score, result.decision, result.risk_band, actual_target)
        st.divider()

        col_left, col_right = st.columns([1.05, 1.25])
        with col_left:
            st.subheader("Top 5 Business Reason Codes")
            display_rc = format_reason_table(result.reason_codes)
            st.dataframe(
                display_rc.style.format({"Net SHAP": "{:+.4f}"}),
                width="stretch",
                hide_index=True,
            )

            with st.expander("View Raw Top-15 Feature Values & SHAP"):
                raw_view = result.feature_shap[["feature", "value", "shap_value"]].head(15).copy()
                raw_view["value"] = raw_view["value"].map(format_value)
                st.dataframe(
                    raw_view.style.format({"shap_value": "{:+.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        with col_right:
            st.subheader("Local SHAP Feature Breakdown")
            st.pyplot(plot_shap_bar(result.feature_shap, top_n=12))

    with tab2:
        st.markdown(
            "Adjust key risk drivers for the selected applicant to observe real-time changes in **PD** and **SHAP**."
        )
        sim_row = applicant_row.copy()

        def _safe_float(col: str, default: float) -> float:
            if col in sim_row.columns and pd.notna(sim_row[col].iloc[0]):
                return float(sim_row[col].iloc[0])
            return default

        c1, c2, c3 = st.columns(3)
        ext_mean = c1.slider(
            "External Bureau Score (EXT_SOURCE_MEAN)",
            min_value=0.02,
            max_value=0.90,
            value=float(np.clip(_safe_float("EXT_SOURCE_MEAN", 0.50), 0.02, 0.90)),
            step=0.02,
        )
        amt_credit = c1.number_input(
            "Loan Amount (AMT_CREDIT)",
            min_value=50_000.0,
            max_value=2_500_000.0,
            value=float(np.clip(_safe_float("AMT_CREDIT", 500_000.0), 50_000.0, 2_500_000.0)),
            step=25_000.0,
        )

        amt_annuity = c2.number_input(
            "Annual Payment (AMT_ANNUITY)",
            min_value=5_000.0,
            max_value=200_000.0,
            value=float(np.clip(_safe_float("AMT_ANNUITY", 25_000.0), 5_000.0, 200_000.0)),
            step=2_500.0,
        )
        cc_util = c2.slider(
            "Max CC Utilization 12M (CC_12M_UTILIZATION_MAX)",
            min_value=0.0,
            max_value=1.50,
            value=float(np.clip(_safe_float("CC_12M_UTILIZATION_MAX", 0.30), 0.0, 1.50)),
            step=0.05,
        )

        prev_refused = c3.slider(
            "Previous Refusal Ratio (PREV_IS_REFUSED_MEAN)",
            min_value=0.0,
            max_value=1.0,
            value=float(np.clip(_safe_float("PREV_IS_REFUSED_MEAN", 0.0), 0.0, 1.0)),
            step=0.05,
        )
        bureau_debt = c3.slider(
            "Max Bureau Debt Ratio (BUREAU_DEBT_RATIO_MAX)",
            min_value=0.0,
            max_value=1.50,
            value=float(np.clip(_safe_float("BUREAU_DEBT_RATIO_MAX", 0.40), 0.0, 1.50)),
            step=0.05,
        )

        for col, val in [
            ("EXT_SOURCE_MEAN", ext_mean),
            ("EXT_SOURCE_MIN", min(ext_mean, _safe_float("EXT_SOURCE_MIN", ext_mean))),
            ("EXT_SOURCE_MAX", max(ext_mean, _safe_float("EXT_SOURCE_MAX", ext_mean))),
            ("AMT_CREDIT", amt_credit),
            ("AMT_ANNUITY", amt_annuity),
            ("CC_12M_UTILIZATION_MAX", cc_util),
            ("PREV_IS_REFUSED_MEAN", prev_refused),
            ("BUREAU_DEBT_RATIO_MAX", bureau_debt),
        ]:
            if col in sim_row.columns:
                sim_row[col] = val

        if "CREDIT_TERM" in sim_row.columns and amt_credit > 0:
            sim_row["CREDIT_TERM"] = amt_annuity / amt_credit
        if "AMT_GOODS_PRICE" in sim_row.columns and pd.notna(sim_row["AMT_GOODS_PRICE"].iloc[0]):
            goods = float(sim_row["AMT_GOODS_PRICE"].iloc[0])
            if goods > 0 and "CREDIT_GOODS_RATIO" in sim_row.columns:
                sim_row["CREDIT_GOODS_RATIO"] = amt_credit / goods
            if "GOODS_PRICE_DIFF" in sim_row.columns:
                sim_row["GOODS_PRICE_DIFF"] = amt_credit - goods

        sim_result = scorer.score_applicant(sim_row, top_k=5)
        st.divider()
        render_decision_header(sim_result.pd_score, sim_result.decision, sim_result.risk_band)

        s_left, s_right = st.columns([1.05, 1.25])
        with s_left:
            st.subheader("Simulated Reason Codes")
            sim_rc = format_reason_table(sim_result.reason_codes)
            st.dataframe(
                sim_rc.style.format({"Net SHAP": "{:+.4f}"}),
                width="stretch",
                hide_index=True,
            )
        with s_right:
            st.subheader("Simulated SHAP Drivers")
            st.pyplot(plot_shap_bar(sim_result.feature_shap, top_n=10))

    with tab3:
        st.subheader("Model Progression (5-Fold Stratified CV)")
        metrics_df = pd.DataFrame(
            [
                {"Iteration": "v1_app_only", "Model": "LogReg (optbinning)", "ROC-AUC": 0.75132, "Gini": 0.50265, "KS": 0.37402, "Brier": 0.06837, "Features": 39},
                {"Iteration": "v1_app_only", "Model": "LightGBM", "ROC-AUC": 0.76856, "Gini": 0.53712, "KS": 0.40082, "Brier": 0.06727, "Features": 141},
                {"Iteration": "v1_app_only", "Model": "CatBoost", "ROC-AUC": 0.76931, "Gini": 0.53862, "KS": 0.40167, "Brier": 0.06723, "Features": 141},
                {"Iteration": "v2_relational", "Model": "LogReg (optbinning)", "ROC-AUC": 0.77629, "Gini": 0.55257, "KS": 0.41554, "Brier": 0.06679, "Features": 112},
                {"Iteration": "v2_relational", "Model": "LightGBM", "ROC-AUC": 0.79353, "Gini": 0.58706, "KS": 0.44395, "Brier": 0.06546, "Features": 322},
                {"Iteration": "v2_relational", "Model": "CatBoost", "ROC-AUC": 0.79390, "Gini": 0.58781, "KS": 0.44557, "Brier": 0.06544, "Features": 322},
                {"Iteration": "v3_tuned", "Model": "LightGBM (Optuna, Top 220)", "ROC-AUC": 0.79574, "Gini": 0.59148, "KS": 0.44800, "Brier": 0.06526, "Features": 220},
                {"Iteration": "v3_final", "Model": "Blend (63% LGBM v3 + 37% CB v2)", "ROC-AUC": 0.79665, "Gini": 0.59331, "KS": 0.45014, "Brier": 0.06519, "Features": 322},
            ]
        )
        st.dataframe(
            metrics_df.style.format({"ROC-AUC": "{:.5f}", "Gini": "{:.5f}", "KS": "{:.5f}", "Brier": "{:.5f}"}),
            width="stretch",
            hide_index=True,
        )

        st.divider()
        cal_img = FIG_DIR / "calibration_and_deciles.png"
        shap_img = FIG_DIR / "shap_beeswarm_top20.png"

        if cal_img.exists():
            st.subheader("Calibration Curve & Risk Decile Lift (Top Decile Lift: 3.88x)")
            st.image(str(cal_img), width="stretch")

        if shap_img.exists():
            st.subheader("Global SHAP Feature Impact (Top 20 Drivers)")
            st.image(str(shap_img), width="stretch")


if __name__ == "__main__":
    main()