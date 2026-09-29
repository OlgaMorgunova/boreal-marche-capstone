"""FastAPI service. Models and rules are loaded once at startup (lifespan), never per request.

``/health`` = the process is alive (liveness). ``/ready`` = artifacts are loaded and requests can be
served (readiness; returns 503 otherwise so a load balancer keeps traffic away).
Still to come in D4: ``/chat`` (Head C), ``/metrics`` and prediction logging to a database table.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status

from boreal_marche import __version__
from boreal_marche.config import Paths, load_params
from boreal_marche.features import FEATURES
from boreal_marche.recommend.engine import RuleEngine
from boreal_marche.registry import load_retention_model
from boreal_marche.retention.artifact import RetentionModel
from boreal_marche.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
    VersionResponse,
)
from boreal_marche.utils import git_commit

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    params, paths = load_params(), Paths()
    STATE.clear()
    STATE["errors"] = []
    try:
        STATE["model"], STATE["model_source"] = load_retention_model(params, paths)
    except Exception as exc:  # noqa: BLE001 - reported by /ready
        STATE["errors"].append(f"model: {exc}")
    try:
        STATE["engine"] = RuleEngine.from_json(paths.rules_file, min_lift=params.recommend.min_lift)
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"rules: {exc}")
    yield
    STATE.clear()


app = FastAPI(title="Boréal Marché API", version=__version__, lifespan=lifespan)


def _model() -> RetentionModel:
    m = STATE.get("model")
    if m is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "retention model not loaded")
    return m  # type: ignore[no-any-return]


def _engine() -> RuleEngine:
    e = STATE.get("engine")
    if e is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "rules not loaded")
    return e  # type: ignore[no-any-return]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready(response: Response) -> ReadyResponse:
    ok_model, ok_rules = "model" in STATE, "engine" in STATE
    if not (ok_model and ok_rules):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        ready=ok_model and ok_rules,
        model_loaded=ok_model,
        rules_loaded=ok_rules,
        errors=list(STATE.get("errors", [])),
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    model = STATE.get("model")
    engine = STATE.get("engine")
    return VersionResponse(
        api_version=__version__,
        git_commit=os.environ.get("GIT_SHA") or git_commit(),
        model_version=model.version if model else None,
        model_source=STATE.get("model_source"),
        rules_mined_at=engine.header.get("mined_at") if engine else None,
        rule_count=len(engine.rules) if engine else None,
    )


def _predict(reqs: list[PredictRequest]) -> list[PredictResponse]:
    model = _model()
    X = pd.DataFrame([r.features.model_dump() for r in reqs])[FEATURES]
    X = X.astype({c: "float64" for c in X.columns if c not in ("region", "season_q")})
    p = model.predict_proba(X)[:, 1]
    return [
        PredictResponse(
            customer_ref=r.customer_ref,
            p_return=float(pi),
            send_coupon=bool(pi <= model.threshold),
            threshold=model.threshold,
            model_version=model.version,
        )
        for r, pi in zip(reqs, p, strict=True)
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return _predict([req])[0]


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    return BatchPredictResponse(predictions=_predict(req.items))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    engine = _engine()
    res = engine.recommend(req.basket, k=req.k)
    return RecommendResponse(
        strategy=res.strategy,
        unknown_items=res.unknown_items,
        rules_mined_at=engine.header.get("mined_at"),
        recommendations=[
            RecommendedItem(
                item=r.item,
                description=r.description,
                source=r.source,
                because=sorted(r.rule.antecedent) if r.rule else None,
                confidence=r.rule.confidence if r.rule else None,
                lift=r.rule.lift if r.rule else None,
            )
            for r in res.recommendations
        ],
    )
