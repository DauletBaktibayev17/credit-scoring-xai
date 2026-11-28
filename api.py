from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PROCESSED_TRAIN_V2_FILE
from src.scorer import CreditScorer

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = CreditScorer()
    df = pd.read_parquet(PROCESSED_TRAIN_V2_FILE)
    state["portfolio"] = df.set_index("SK_ID_CURR", drop=False)
    yield
    state.clear()


app = FastAPI(
    title="Credit Scoring XAI API",
    description="Real-time PD prediction and SHAP Adverse Action Reason Codes (5-Fold Tuned LightGBM).",
    version="1.0.0",
    lifespan=lifespan,
)


class ReasonCode(BaseModel):
    reason_category: str
    net_shap_impact: float
    direction: str


class ScoreResponse(BaseModel):
    sk_id_curr: int
    pd_score: float
    decision: str
    risk_band: str
    top_reason_codes: list[ReasonCode]


class WhatIfRequest(BaseModel):
    sk_id_curr: int = Field(..., examples=[395476])
    overrides: dict[str, float] = Field(
        default_factory=dict,
        examples=[{"EXT_SOURCE_MEAN": 0.65, "CC_12M_UTILIZATION_MAX": 0.2}],
    )
    top_k: int = Field(default=5, ge=1, le=10)


def _build_response(sk_id_curr: int, row: pd.DataFrame, top_k: int = 5) -> ScoreResponse:
    scorer: CreditScorer = state["scorer"]
    res = scorer.score_applicant(row, top_k=top_k)
    reasons = [
        ReasonCode(
            reason_category=r.reason_group,
            net_shap_impact=round(float(r.shap_value), 4),
            direction=r.direction,
        )
        for r in res.reason_codes.itertuples()
    ]
    return ScoreResponse(
        sk_id_curr=sk_id_curr,
        pd_score=round(res.pd_score, 5),
        decision=res.decision,
        risk_band=res.risk_band,
        top_reason_codes=reasons,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    scorer: CreditScorer = state["scorer"]
    return {
        "status": "ok",
        "model": "lgbm_v3_tuned",
        "n_folds": len(scorer.folds),
        "n_features": len(scorer.feature_cols),
    }


@app.get("/score/{sk_id_curr}", response_model=ScoreResponse)
def score_by_id(sk_id_curr: int, top_k: int = 5) -> ScoreResponse:
    portfolio: pd.DataFrame = state["portfolio"]
    if sk_id_curr not in portfolio.index:
        raise HTTPException(status_code=404, detail=f"Applicant SK_ID_CURR={sk_id_curr} not found.")
    row = portfolio.loc[[sk_id_curr]].copy()
    return _build_response(sk_id_curr, row, top_k=top_k)


@app.post("/score", response_model=ScoreResponse)
def score_with_overrides(req: WhatIfRequest) -> ScoreResponse:
    portfolio: pd.DataFrame = state["portfolio"]
    if req.sk_id_curr not in portfolio.index:
        raise HTTPException(status_code=404, detail=f"Applicant SK_ID_CURR={req.sk_id_curr} not found.")

    row = portfolio.loc[[req.sk_id_curr]].copy()
    for col, val in req.overrides.items():
        if col in row.columns:
            row[col] = val

    return _build_response(req.sk_id_curr, row, top_k=req.top_k)