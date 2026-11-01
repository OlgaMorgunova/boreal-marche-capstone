"""Pydantic v2 request/response contracts. Extra fields are rejected on every request model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from boreal_marche.reskin import REGIONS


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CustomerFeatures(Strict):
    """Point-in-time feature vector (see reports/feature_spec.md). Nullable cadence features are
    ``None`` when undefined (fewer than two order days) — never silently zero.

    Since v0.9.5 the model uses 19 of these (D5 data-minimisation audit). The other fields in
    ``features.EXCLUDED_FROM_MODEL`` are optional: still accepted so existing clients keep working,
    never stored. ``region`` stays required (delivery, monitoring, fairness audit)."""

    recency_days: float = Field(ge=0)
    tenure_days: float = Field(ge=0)
    tenure_censored: int = Field(ge=0, le=1)
    orders_30d: int = Field(ge=0)
    orders_90d: int = Field(ge=0)
    orders_180d: int = Field(ge=1)
    orders_lifetime: int | None = Field(default=None, ge=1)
    active_months_180d: int = Field(ge=1, le=7)
    revenue_30d: float = Field(ge=0)
    revenue_90d: float = Field(ge=0)
    revenue_180d: float = Field(gt=0)
    revenue_lifetime: float | None = Field(default=None, gt=0)
    aov_180d: float = Field(gt=0)
    max_order_value_180d: float = Field(gt=0)
    mean_unit_price_180d: float = Field(gt=0)
    mean_gap_days: float | None = Field(default=None, ge=0)
    std_gap_days: float | None = Field(default=None, ge=0)
    overdue_ratio: float | None = Field(default=None, ge=0)
    orders_trend: float
    revenue_trend: float | None = Field(default=None, ge=-1, le=1)
    lines_per_order_180d: float | None = Field(default=None, ge=1)
    units_per_line_180d: float | None = Field(default=None, gt=0)
    distinct_skus_180d: int | None = Field(default=None, ge=1)
    sku_repeat_rate_180d: float | None = Field(default=None, ge=0, le=1)
    share_seasonal: float | None = Field(default=None, ge=0, le=1)
    share_kitchen: float | None = Field(default=None, ge=0, le=1)
    share_home_decor: float | None = Field(default=None, ge=0, le=1)
    share_bags_wrap: float | None = Field(default=None, ge=0, le=1)
    share_lighting: float | None = Field(default=None, ge=0, le=1)
    category_entropy_180d: float | None = Field(default=None, ge=0)
    cancel_lines_180d: int | None = Field(default=None, ge=0)
    cancel_rate_180d: float | None = Field(default=None, ge=0, le=1)
    weekend_share_180d: float | None = Field(default=None, ge=0, le=1)
    mean_order_hour_180d: float | None = Field(default=None, ge=0, le=23)
    outlier_line_share_180d: float | None = Field(default=None, ge=0, le=1)
    region: Literal[tuple(REGIONS)]  # type: ignore[valid-type]
    season_q: Literal["1", "2", "3", "4"]


class PredictRequest(Strict):
    customer_ref: str | None = Field(
        default=None,
        max_length=64,
        description="Opaque reference echoed back; not used by the model",
    )
    features: CustomerFeatures


class Factor(BaseModel):
    """One principal factor of a prediction: exact additive contribution (TreeSHAP) to the
    calibrated log-odds of returning. Explains the model, not the customer's motives."""

    feature: str
    label: str = Field(description="Plain definition of the feature")
    value: float | str | None = Field(description="The customer's value (None = undefined)")
    effect: float = Field(description="Contribution to the log-odds of returning")
    direction: Literal["raises P(return)", "lowers P(return)"]


class PredictResponse(BaseModel):
    customer_ref: str | None
    p_return: float = Field(description="Calibrated P(order within 90 days)")
    send_coupon: bool = Field(description="Proposed action; a human approves the campaign list")
    threshold: float
    model_version: str
    top_factors: list[Factor] | None = Field(
        default=None, description="Principal factors, largest first (only with ?explain=N)"
    )
    action: Literal["coupon", "holdout", "review", "none"] | None = Field(
        default=None,
        description="Campaign routing (D7): coupon = proposed; holdout = proposed but withheld "
        "this campaign to measure the save rate; review = not proposed, listed for a human "
        "(high-value review list); none = not proposed. None when no campaign policy is loaded.",
    )
    spend_quartile: str | None = Field(
        default=None, description="Spend quartile of revenue_180d (training edges)"
    )


class CampaignEntry(BaseModel):
    ref: str
    region: str
    p_return: float
    spend_quartile: str
    revenue_180d: float
    action: Literal["coupon", "holdout", "review", "none"]


class CampaignResponse(BaseModel):
    """The campaign list a manager approves, scored on the demo customers (not logged)."""

    campaign_id: str
    model_version: str
    threshold: float
    review_thresholds: dict[str, float]
    holdout_share: float
    counts: dict[str, int]
    review_list: list[CampaignEntry]
    holdout: list[CampaignEntry]


class BatchPredictRequest(Strict):
    items: list[PredictRequest] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class RecommendRequest(Strict):
    basket: list[str] = Field(default_factory=list, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


class RecommendedItem(BaseModel):
    item: str
    description: str
    source: Literal["rule", "popularity"]
    because: list[str] | None = Field(description="Antecedent of the firing rule")
    confidence: float | None
    lift: float | None


class RecommendResponse(BaseModel):
    strategy: Literal["rules", "rules+popularity", "popularity_fallback", "empty_basket"]
    recommendations: list[RecommendedItem]
    unknown_items: list[str]
    rules_mined_at: str | None


class VersionResponse(BaseModel):
    api_version: str
    git_commit: str
    image_tag: str | None
    model_version: str | None
    model_source: str | None
    rules_mined_at: str | None
    rule_count: int | None


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    rules_loaded: bool
    assistant_loaded: bool
    errors: list[str]
    startup_ms: dict[str, float] = Field(
        default_factory=dict, description="Load time per artifact, warm-up and total (D5)"
    )


# ---------------------------------------------------------------- customers / products (D4) ---


class CustomerSummary(BaseModel):
    ref: str
    region: str
    orders_180d: int = Field(description="Orders in the last 180 days")
    recency_days: float


class CustomerList(BaseModel):
    customers: list[CustomerSummary]
    source: str


class ProductOut(BaseModel):
    sku: str
    description: str
    category: str
    median_price: float
    n_orders: int


class ProductList(BaseModel):
    query: str
    products: list[ProductOut]


# ------------------------------------------------------------------------ assistant (Head C) ---

ProviderName = Literal["rule", "ollama", "openai"]


class ChatRequest(Strict):
    message: str = Field(min_length=1, max_length=1000)
    provider: ProviderName = "rule"
    customer_ref: str | None = Field(
        default=None,
        pattern=r"^BM-[0-9A-F]{8}$",
        description="Signed-in customer; the retention tool only ever scores this customer",
    )
    lang: Literal["en", "fr"] | None = Field(
        default=None, description="Answer language; detected from the message when omitted"
    )


class Citation(BaseModel):
    kind: Literal["policy", "tool"]
    ref: str = Field(description="Policy chunk id, or the API endpoint a tool called")
    title: str
    section: str | None = None
    score: float | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    lang: Literal["en", "fr"]
    provider: ProviderName
    tool: str
    refused: bool
    refusal_reason: Literal["out_of_scope", "privacy"] | None = None
    handoff: bool = Field(description="Passed to a human (no grounding, or a tool failed)")
    citations: list[Citation]
    tool_result: dict[str, Any] | None = None
    guardrail_flags: list[str]
    latency_ms: float
    usage: ChatUsage


class ProviderStatus(BaseModel):
    name: ProviderName
    available: bool
    detail: str
    model: str | None
