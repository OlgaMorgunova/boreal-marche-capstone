"""FastAPI service. Models, rules, the retrieval index and the reference profile are loaded once at
startup (lifespan), never per request: the two slow loads (retention model, embedding model) run
in parallel, then one warm-up call per head runs before ``/ready`` turns 200, so the first customer
does not pay for lazy initialisation. Load and warm-up times are reported by ``/ready`` (D5 §2).

``/health`` = the process is alive (liveness). ``/ready`` = every artifact is loaded and requests
can be served (readiness; 503 otherwise, so a load balancer keeps traffic away).

Every request is timed and written to the log store with its status (structured JSON line on stdout
as well); every prediction, recommendation and chat turn is written to its own table. A background
thread evaluates ``/monitoring/report`` every ``monitoring.check_interval_seconds`` and raises
alerts on level changes (``monitoring/alerts.py``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.routing import Match

from boreal_marche import __version__
from boreal_marche.assistant.guardrails import fold
from boreal_marche.assistant.index import load_retriever
from boreal_marche.assistant.providers import build_providers
from boreal_marche.assistant.service import Assistant, ProviderUnavailable
from boreal_marche.assistant.tools import Toolbox
from boreal_marche.config import Paths, load_params
from boreal_marche.features import CATEGORICAL, COLLECTED
from boreal_marche.monitoring import alerts
from boreal_marche.monitoring.drift import STATUS_ORDER, build_report, realised_metrics, timeseries
from boreal_marche.monitoring.store import LogStore
from boreal_marche.recommend.engine import RuleEngine
from boreal_marche.registry import load_retention_model
from boreal_marche.retention.artifact import RetentionModel
from boreal_marche.retention.campaign import CampaignPolicy
from boreal_marche.retention.explain import top_factors
from boreal_marche.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CampaignEntry,
    CampaignResponse,
    ChatRequest,
    ChatResponse,
    CustomerFeatures,
    CustomerList,
    CustomerSummary,
    Factor,
    PredictRequest,
    PredictResponse,
    ProductList,
    ProductOut,
    ProviderStatus,
    ReadyResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
    VersionResponse,
)
from boreal_marche.utils import git_commit, read_json

STATE: dict[str, Any] = {}
log = logging.getLogger("boreal.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "boreal_requests_total", "HTTP requests", ["method", "path", "status"], registry=REGISTRY
)
LATENCY = Histogram(
    "boreal_request_latency_seconds",
    "Server-side request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)
PREDICTIONS = Counter(
    "boreal_predictions_total", "Retention predictions", ["send_coupon"], registry=REGISTRY
)
ACTIONS = Counter(
    "boreal_campaign_action_total", "Campaign routing of predictions", ["action"],
    registry=REGISTRY,
)  # fmt: skip
STRATEGIES = Counter(
    "boreal_recommend_strategy_total", "Recommendation strategy fired", ["strategy"],
    registry=REGISTRY,
)  # fmt: skip
CHATS = Counter("boreal_chat_total", "Assistant turns", ["provider", "tool"], registry=REGISTRY)
CHECK_STATUS = Gauge(
    "boreal_monitoring_status",
    "Traffic light per check: -1 insufficient data, 0 green, 1 amber, 2 red",
    ["check"],
    registry=REGISTRY,
)
FEATURE_PSI = Gauge("boreal_feature_psi", "PSI per input feature", ["feature"], registry=REGISTRY)


def _load(name: str, fn: Callable[[], Any]) -> None:
    t0 = time.perf_counter()
    try:
        STATE[name] = fn()
    except Exception as exc:  # noqa: BLE001 - reported by /ready
        STATE["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    STATE["startup_ms"][name] = round(1000 * (time.perf_counter() - t0), 1)


def _campaign(paths: Paths) -> CampaignPolicy:
    """Frozen review thresholds belong to one model: a stale policy is refused (reported by /ready,
    predictions then carry ``action=None``) rather than applied to a model it was not fitted on."""
    policy = CampaignPolicy.load(paths.campaign_policy)
    served = STATE.get("model")
    if served is not None and served.version != policy.model_version:
        raise ValueError(f"policy is for {policy.model_version}, model is {served.version}")
    return policy


def _warm_up() -> None:
    """One call per head on a committed demo customer: XGBoost, ONNX Runtime and pandas initialise
    lazily on first use, which otherwise lands on the first real request."""
    if "customers" not in STATE:
        return
    feats = next(iter(STATE["customers"].values()))["features"]
    if "model" in STATE:
        top_factors(STATE["model"], _frame([feats], STATE["model"].feature_names), 3)
    if "engine" in STATE:
        STATE["engine"].recommend([], k=5)
    if "assistant" in STATE:
        STATE["assistant"].toolbox.retriever.search("returns policy", 1)


def _monitor_loop(stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        try:
            refresh_report()
        except Exception as exc:  # noqa: BLE001 - the loop must survive a bad evaluation
            log.warning(json.dumps({"event": "monitor_failed", "error": str(exc)[:200]}))


def refresh_report() -> dict[str, Any]:
    params = STATE["params"]
    store: LogStore = STATE["store"]
    store.flush()
    report = build_report(store, STATE["profile"], params.monitoring)
    report["ground_truth"] = realised_metrics(store, params.windows.label_days)
    report["new_alerts"] = alerts.evaluate(store, report)
    for c in report["checks"]:
        CHECK_STATUS.labels(c["name"]).set(STATUS_ORDER[c["status"]])
    for f in report["features"]:
        FEATURE_PSI.labels(f["feature"]).set(f["psi"])
    STATE["last_report"] = report
    return report


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    params, paths = load_params(), Paths()
    STATE.clear()
    # Resolved once: /version must not shell out to git on every request.
    STATE.update(
        params=params, paths=paths, errors=[], startup_ms={},
        git_commit=os.environ.get("GIT_SHA") or git_commit(),
    )  # fmt: skip
    t_start = time.perf_counter()

    def _model() -> RetentionModel:
        model, STATE["model_source"] = load_retention_model(params, paths)
        return model

    def _catalogue() -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = read_json(paths.catalogue)["products"]
        for p in products:  # folded once here, not for every product on every /products call
            p["_folded"] = fold(p["description"])
        return products

    STATE["store"] = LogStore(os.environ.get("DATABASE_URL", "sqlite:///./boreal.db"))
    STATE["api_client"] = httpx.Client(
        base_url=os.environ.get("BOREAL_API_URL", "http://127.0.0.1:8000"), timeout=10
    )
    a = params.assistant

    def _assistant() -> Assistant:
        toolbox = Toolbox(load_retriever(params, paths), STATE["api_client"],
                          a.similarity_floor, a.top_k)  # fmt: skip
        providers = build_providers(a.ollama_model, a.openai_model, a.request_timeout_seconds)
        price = (a.openai_price_per_mtok.input, a.openai_price_per_mtok.output)
        return Assistant(toolbox, providers, price)

    def _engine() -> RuleEngine:
        return RuleEngine.from_json(paths.rules_file, min_lift=params.recommend.min_lift)

    def _customers() -> dict[str, Any]:
        return {c["ref"]: c for c in read_json(paths.demo_customers)["customers"]}

    # The retention model and the embedding model (ONNX session) are the two slow, independent
    # loads. Measured (D5 §2): side by side saves ~0.45 s when the model comes from the MLflow
    # registry (I/O-bound), nothing measurable from the local file (~0.1 s, the ONNX session
    # dominates).
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = [pool.submit(_load, "model", _model), pool.submit(_load, "assistant", _assistant)]
        _load("engine", _engine)
        _load("customers", _customers)
        _load("catalogue", _catalogue)
        _load("profile", lambda: read_json(paths.reference_profile))
        for f in slow:
            f.result()
    _load("campaign", lambda: _campaign(paths))
    _load("warm_up", _warm_up)
    STATE["startup_ms"]["total"] = round(1000 * (time.perf_counter() - t_start), 1)
    log.info(json.dumps({"event": "startup", "ms": STATE["startup_ms"]}))
    stop = threading.Event()
    if os.environ.get("BOREAL_MONITOR_LOOP", "1") == "1" and "profile" in STATE:
        threading.Thread(
            target=_monitor_loop,
            args=(stop, params.monitoring.check_interval_seconds),
            name="monitor",
            daemon=True,
        ).start()
    yield
    stop.set()
    STATE["store"].close()
    STATE.clear()


app = FastAPI(
    title="Boréal Marché API",
    version=__version__,
    lifespan=lifespan,
    description="Retention prediction (Head A), basket recommendations (Head B), grounded "
    "bilingual assistant (Head C) and live monitoring.",
)


def _route_path(request: Request) -> str:
    """Route template (``/customers/{ref}/features``), never the raw path: bounded label space."""
    for route in request.app.routes:
        if route.matches(request.scope)[0] == Match.FULL:
            return str(getattr(route, "path", request.url.path))
    return "unmatched"


@app.middleware("http")
async def observe(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    cohort = re.sub(r"[^a-z_]", "", request.headers.get("X-Traffic-Cohort", "").lower())[:16]
    request.state.request_id, request.state.cohort = request_id, cohort or None
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(json.dumps({"event": "unhandled", "request_id": request_id}))
        response = JSONResponse(
            {"detail": "internal error", "request_id": request_id}, status_code=500
        )
    ms = 1000 * (time.perf_counter() - t0)
    path = _route_path(request)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(ms / 1000)
    response.headers["X-Request-ID"] = request_id
    if path not in ("/metrics", "/health"):
        row = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "latency_ms": round(ms, 3),
            "caller": request.headers.get("X-Internal-Caller", "external")[:16],
            "cohort": request.state.cohort,
        }
        log.info(json.dumps({"event": "request", **row}))
        if "store" in STATE:
            STATE["store"].log("request_log", row)
    return response


def _need(key: str, what: str) -> Any:
    obj = STATE.get(key)
    if obj is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{what} not loaded")
    return obj


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(response: Response) -> ReadyResponse:
    ok = {k: k in STATE for k in ("model", "engine", "assistant", "customers", "profile")}
    if not all(ok.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        ready=all(ok.values()),
        model_loaded=ok["model"],
        rules_loaded=ok["engine"],
        assistant_loaded=ok["assistant"],
        errors=list(STATE.get("errors", [])),
        startup_ms=dict(STATE.get("startup_ms", {})),
    )


@app.get("/version", response_model=VersionResponse, tags=["ops"])
def version() -> VersionResponse:
    model = STATE.get("model")
    engine = STATE.get("engine")
    return VersionResponse(
        api_version=__version__,
        git_commit=STATE.get("git_commit") or "unknown",
        image_tag=os.environ.get("IMAGE_TAG") or None,
        model_version=model.version if model else None,
        model_source=STATE.get("model_source"),
        rules_mined_at=engine.header.get("mined_at") if engine else None,
        rule_count=len(engine.rules) if engine else None,
    )


@app.get("/metrics", tags=["ops"], response_class=Response)
def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------- Head A ---


def _frame(rows: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    """Model input built column by column: the same frame as ``pd.DataFrame(rows)[features]`` with
    numeric columns cast to float64 (asserted in tests), ~8x faster for one row (D5 §2)."""
    return pd.DataFrame(
        {
            f: np.array([r[f] for r in rows], dtype=object if f in CATEGORICAL else "float64")
            for f in features
        },
        columns=features,
    )


def _predict(
    reqs: list[PredictRequest], request: Request, explain: int = 0
) -> list[PredictResponse]:
    model: RetentionModel = _need("model", "retention model")
    X = _frame([r.features.model_dump() for r in reqs], model.feature_names)
    p = model.predict_proba(X)[:, 1]
    factors = top_factors(model, X, explain) if explain else [None] * len(reqs)
    policy: CampaignPolicy | None = STATE.get("campaign")
    out = []
    for r, pi, fs in zip(reqs, p, factors, strict=True):
        send = bool(pi <= model.threshold)
        spend = r.features.revenue_180d
        action = policy.action(float(pi), spend, r.customer_ref) if policy else None
        out.append(
            PredictResponse(
                customer_ref=r.customer_ref,
                p_return=float(pi),
                send_coupon=send,
                threshold=model.threshold,
                model_version=model.version,
                top_factors=[Factor(**f) for f in fs] if fs is not None else None,
                action=action,
                spend_quartile=policy.spend_quartile(spend) if policy else None,
            )
        )
        PREDICTIONS.labels(str(send).lower()).inc()
        if action:
            ACTIONS.labels(action).inc()
        STATE["store"].log(
            "prediction_log",
            {
                "request_id": request.state.request_id,
                "customer_ref": r.customer_ref,
                "model_version": model.version,
                "p_return": float(pi),
                "send_coupon": send,
                "threshold": model.threshold,
                # Only what the model uses, plus region (D5 data minimisation).
                "features": r.features.model_dump_json(include=set(COLLECTED)),
                "cohort": request.state.cohort,
            },
        )
    return out


EXPLAIN = Query(
    0, ge=0, le=10, description="Also return the N principal factors of each prediction (s. 12.1)"
)


@app.post("/predict", response_model=PredictResponse, tags=["retention"])
def predict(req: PredictRequest, request: Request, explain: int = EXPLAIN) -> PredictResponse:
    return _predict([req], request, explain)[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["retention"])
def predict_batch(
    req: BatchPredictRequest, request: Request, explain: int = EXPLAIN
) -> BatchPredictResponse:
    return BatchPredictResponse(predictions=_predict(req.items, request, explain))


@app.get("/campaign", response_model=CampaignResponse, tags=["retention"])
def campaign() -> CampaignResponse:
    """This quarter's campaign list for the demo customers, as the reviewer sees it: proposed
    coupons, the hold-out and the high-value review list. Scored, not logged (no drift traffic)."""
    model: RetentionModel = _need("model", "retention model")
    policy: CampaignPolicy = _need("campaign", "campaign policy")
    data: dict[str, Any] = _need("customers", "demo customers")
    rows = list(data.values())
    p = model.predict_proba(_frame([c["features"] for c in rows], model.feature_names))[:, 1]
    entries = [
        CampaignEntry(
            ref=c["ref"],
            region=c["region"],
            p_return=float(pi),
            spend_quartile=policy.spend_quartile(c["features"]["revenue_180d"]),
            revenue_180d=c["features"]["revenue_180d"],
            action=policy.action(float(pi), c["features"]["revenue_180d"], c["ref"]),
        )
        for c, pi in zip(rows, p, strict=True)
    ]
    counts = {
        a: sum(e.action == a for e in entries) for a in ("coupon", "holdout", "review", "none")
    }
    return CampaignResponse(
        campaign_id=policy.campaign_id,
        model_version=policy.model_version,
        threshold=policy.threshold,
        review_thresholds=policy.review_thresholds,
        holdout_share=policy.holdout_share,
        counts=counts,
        review_list=sorted(
            (e for e in entries if e.action == "review"), key=lambda e: -e.revenue_180d
        ),
        holdout=[e for e in entries if e.action == "holdout"],
    )


@app.get("/customers", response_model=CustomerList, tags=["retention"])
def customers(limit: int = Query(50, ge=1, le=300)) -> CustomerList:
    """Demo customers (pseudonymised validation-snapshot features): the feature-store stand-in."""
    data: dict[str, Any] = _need("customers", "demo customers")
    rows = list(data.values())[:limit]
    return CustomerList(
        source="validation snapshot, pseudonymised (artifacts/demo_customers.json)",
        customers=[
            CustomerSummary(
                ref=c["ref"],
                region=c["region"],
                orders_180d=c["features"]["orders_180d"],
                recency_days=c["features"]["recency_days"],
            )
            for c in rows
        ],
    )


@app.get("/customers/{ref}/features", response_model=CustomerFeatures, tags=["retention"])
def customer_features(ref: str) -> CustomerFeatures:
    data: dict[str, Any] = _need("customers", "demo customers")
    if ref not in data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown customer {ref}")
    return CustomerFeatures.model_validate(data[ref]["features"])


# ---------------------------------------------------------------------------- Head B ---


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
    engine: RuleEngine = _need("engine", "rules")
    res = engine.recommend(req.basket, k=req.k)
    STRATEGIES.labels(res.strategy).inc()
    STATE["store"].log(
        "recommendation_log",
        {
            "request_id": request.state.request_id,
            "basket_size": len(set(req.basket)),
            "n_unknown": len(res.unknown_items),
            "strategy": res.strategy,
            "n_items": len(res.recommendations),
            "cohort": request.state.cohort,
        },
    )
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


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


@app.get("/products", response_model=ProductList, tags=["recommendation"])
def products(
    q: str = Query("", max_length=100), limit: int = Query(10, ge=1, le=50)
) -> ProductList:
    """Catalogue search by SKU or by words of the description (all words must match)."""
    catalogue: list[dict[str, Any]] = _need("catalogue", "catalogue")
    code = q.strip().upper()
    hits = [p for p in catalogue if p["sku"] == code]
    if not hits:
        words = [_singular(w) for w in re.findall(r"[a-z0-9]+", fold(q))]
        hits = [p for p in catalogue if words and all(w in p["_folded"] for w in words)]
        if not q.strip():
            hits = catalogue
    return ProductList(query=q, products=[ProductOut.model_validate(p) for p in hits[:limit]])


# ---------------------------------------------------------------------------- Head C ---


def _assistant(req: ChatRequest) -> Assistant:
    assistant: Assistant = _need("assistant", "assistant")
    ok, detail = assistant.providers[req.provider].available()
    if not ok:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"provider '{req.provider}' unavailable: {detail}"
        )
    return assistant


def _log_chat(resp: ChatResponse, request: Request) -> None:
    CHATS.labels(resp.provider, resp.tool).inc()
    STATE["store"].log(
        "chat_log",
        {
            "request_id": request.state.request_id,
            "provider": resp.provider,
            "tool": resp.tool,
            "lang": resp.lang,
            "refused": resp.refused,
            "handoff": resp.handoff,
            "latency_ms": resp.latency_ms,
            "cohort": request.state.cohort,
        },
    )


@app.get("/chat/providers", response_model=list[ProviderStatus], tags=["assistant"])
def chat_providers() -> list[ProviderStatus]:
    assistant: Assistant = _need("assistant", "assistant")
    return assistant.status()


@app.post("/chat", response_model=ChatResponse, tags=["assistant"])
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    try:
        resp = _assistant(req).answer(req)
    except ProviderUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    _log_chat(resp, request)
    return resp


@app.post("/chat/stream", tags=["assistant"])
def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: ``meta``, ``token``..., optional ``replace``, then ``done``."""
    assistant = _assistant(req)

    def events() -> Iterator[str]:
        for ev in assistant.stream(req):
            if ev["type"] == "done":
                _log_chat(ChatResponse.model_validate(ev["response"]), request)
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# ------------------------------------------------------------------------- monitoring ---


@app.get("/monitoring/report", tags=["monitoring"])
def monitoring_report(
    fresh: bool = Query(False, description="Recompute now instead of serving a recent report"),
) -> dict[str, Any]:
    """Drift, data quality and operational checks with traffic lights. A report younger than
    ``monitoring.report_cache_seconds`` is served as is (computing one costs ~100 ms and the
    dashboard polls it); ``fresh=true`` forces a recomputation."""
    _need("profile", "reference profile")
    last = STATE.get("last_report")
    if (
        fresh
        or last is None
        or time.time() - last["generated_at"] > (STATE["params"].monitoring.report_cache_seconds)
    ):
        return refresh_report()
    return dict(last)


@app.get("/monitoring/alerts", tags=["monitoring"])
def monitoring_alerts(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    store: LogStore = STATE["store"]
    df = store.recent("alert_log", limit)
    return json.loads(df.iloc[::-1].to_json(orient="records")) if len(df) else []


@app.get("/monitoring/timeseries", tags=["monitoring"])
def monitoring_timeseries(bucket_seconds: int = Query(60, ge=10, le=3600)) -> list[dict[str, Any]]:
    store: LogStore = STATE["store"]
    store.flush()
    return timeseries(store, bucket_seconds=bucket_seconds)
