"""Typed access to ``params.yaml`` and repository paths.

All pipeline parameters are declared in ``params.yaml`` at the repository root; this module is the
only place that reads it. Modules receive a :class:`Params` object instead of hard-coding values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


def repo_root() -> Path:
    """Return the repository root (overridable with ``BOREAL_ROOT`` for containers/tests)."""
    env = os.environ.get("BOREAL_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataParams(_Frozen):
    source_url: str
    doi: str
    licence: str
    raw_file: str
    expected_sha256: str
    local_candidates: list[str]


class CleaningParams(_Frozen):
    service_stock_codes: list[str]
    outlier_quantile: float


class CutoffParams(_Frozen):
    train: list[str]
    validation: list[str]
    test: list[str]


class WindowParams(_Frozen):
    observation_days: int
    label_days: int
    cutoffs: CutoffParams


class SpotcheckParams(_Frozen):
    n_folds: int


class TuningParams(_Frozen):
    n_trials: int
    timeout_seconds: int


class BusinessParams(_Frozen):
    coupon_cost: float
    discount_rate: float
    save_rate: float
    gross_margin: float
    value_horizon_days: int
    sensitivity_save_rates: list[float]


class RecommendParams(_Frozen):
    holdout_start: str
    validation_start: str
    min_support_grid: list[float]
    min_confidence_grid: list[float]
    min_support: float
    min_confidence: float
    min_lift: float
    max_len: int
    top_k: int


class MlflowParams(_Frozen):
    tracking_uri: str
    experiment_retention: str
    experiment_recommend: str
    registered_model: str


class AwsParams(_Frozen):
    region: str
    instance_type: str
    use_spot: bool
    max_run_seconds: int
    max_wait_seconds: int
    framework_version: str


class AzureParams(_Frozen):
    location: str
    compute_name: str
    vm_size: str
    tier: Literal["low_priority", "dedicated"]
    idle_seconds_before_scale_down: int
    timeout_seconds: int
    experiment: str


class CloudParams(_Frozen):
    provider: Literal["local", "aws", "azure"]
    n_trials: int
    max_parallel: int
    aws: AwsParams
    azure: AzureParams


class TokenPrice(_Frozen):
    input: float
    output: float


class AssistantParams(_Frozen):
    corpus_dir: str
    embedding_model: str
    chunk_max_words: int
    top_k: int
    similarity_floor: float
    ollama_model: str
    openai_model: str
    openai_price_per_mtok: TokenPrice
    request_timeout_seconds: int


class Band(_Frozen):
    """Traffic-light band: value at which a check turns amber, and red."""

    amber: float
    red: float


class MonitoringThresholds(_Frozen):
    psi: Band
    ks: Band
    drifted_feature_share_red: float
    prediction_psi: Band
    missing_rate_delta: Band
    error_rate: Band
    p95_latency_ms: Band
    rule_fire_rate: Band


class MonitoringParams(_Frozen):
    window_predictions: int
    window_requests: int
    min_predictions: int
    check_interval_seconds: int
    reference_sample_rows: int
    exclude_features: list[str]
    thresholds: MonitoringThresholds


class Params(_Frozen):
    seed: int
    data: DataParams
    cleaning: CleaningParams
    windows: WindowParams
    spotcheck: SpotcheckParams
    tuning: TuningParams
    business: BusinessParams
    recommend: RecommendParams
    mlflow: MlflowParams
    cloud: CloudParams
    assistant: AssistantParams
    monitoring: MonitoringParams


@lru_cache(maxsize=4)
def load_params(path: str | None = None) -> Params:
    """Load and validate ``params.yaml`` (cached)."""
    p = Path(path) if path else repo_root() / "params.yaml"
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Params.model_validate(raw)


class Paths:
    """Canonical locations of every data layer, artifact and report."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or repo_root()
        self.raw = self.root / "data" / "raw"
        self.bronze = self.root / "data" / "bronze"
        self.silver = self.root / "data" / "silver"
        self.gold = self.root / "data" / "gold"
        self.manifest = self.root / "data" / "manifest.json"
        self.artifacts = self.root / "artifacts"
        self.reports = self.root / "reports"
        self.figures = self.reports / "figures"
        self.metrics = self.reports / "metrics"

    @property
    def bronze_transactions(self) -> Path:
        return self.bronze / "transactions.parquet"

    @property
    def silver_lines(self) -> Path:
        return self.silver / "lines.parquet"

    @property
    def silver_cancellations(self) -> Path:
        return self.silver / "cancellations.parquet"

    @property
    def model_file(self) -> Path:
        return self.artifacts / "model.joblib"

    @property
    def model_meta(self) -> Path:
        return self.artifacts / "model_meta.json"

    @property
    def rules_file(self) -> Path:
        return self.artifacts / "rules.json"

    @property
    def reference_profile(self) -> Path:
        return self.artifacts / "reference_profile.json"

    @property
    def demo_customers(self) -> Path:
        return self.artifacts / "demo_customers.json"

    @property
    def catalogue(self) -> Path:
        return self.artifacts / "catalogue.json"

    @property
    def index_dir(self) -> Path:
        return self.artifacts / "index"

    def ensure(self) -> None:
        for d in (
            self.raw,
            self.bronze,
            self.silver,
            self.gold,
            self.artifacts,
            self.figures,
            self.metrics,
        ):
            d.mkdir(parents=True, exist_ok=True)
