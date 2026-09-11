from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ConnectionState = Literal["live", "reconnecting", "offline", "mock"]
ModelStatus = Literal["loaded", "loading", "unavailable", "failed"]
ModelScope = Literal["foundation", "finetuned", "research"]
SignalType = Literal["bullish", "bearish", "neutral"]


class BenchmarkSummary(BaseModel):
    accuracy: float | None = None
    pnl_usdt: float | None = None
    compounded_return_pct: float | None = None
    evaluation_window: str | None = None
    lookback: int | None = None
    pred_len: int | None = None
    close_direction_accuracy: float | None = None
    path_direction_accuracy: float | None = None
    candle_direction_accuracy: float | None = None
    majority_baseline_accuracy: float | None = None
    close_mae: float | None = None
    winner_metric_name: str | None = None
    winner_metric_value: float | None = None
    selection_summary: str | None = None


class ModelRecord(BaseModel):
    id: str
    display_name: str
    scope: ModelScope
    family: str
    branch: str
    supported_assets: list[str]
    supported_horizons: list[str]
    status: ModelStatus
    params: str | None = None
    description: str
    execution_mode: str
    checkpoint_path: str | None = None
    runtime_model_id: str | None = None
    runtime_tokenizer_id: str | None = None
    context_length: int | None = None
    default_lookback: int | None = None
    default_pred_len: int | None = None
    supports_live_inference: bool = False
    supports_custom_lookback: bool = False
    supports_custom_pred_len: bool = False
    supports_path_forecast: bool = False
    benchmark: BenchmarkSummary | None = None
    tags: list[str] = Field(default_factory=list)


class ModelFamilyRecord(BaseModel):
    key: str
    title: str
    summary: str
    strengths: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    symbol: str
    last_price: float
    price_change_pct: float
    volume: float
    quote_volume: float
    high_24h: float
    low_24h: float
    open_24h: float
    updated_at: str


class CandlePoint(BaseModel):
    symbol: str
    interval: str
    open_time: str
    close_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


class LiveStatus(BaseModel):
    state: ConnectionState
    detail: str
    updated_at: str


class ForecastPoint(BaseModel):
    index: int
    timestamp: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    price: float
    pct_from_last: float


class PredictionRequest(BaseModel):
    symbol: str
    horizon: str
    model_id: str
    auto_run: bool = False
    lookback: int | None = Field(default=None, ge=1)
    pred_len: int | None = Field(default=None, ge=1)


class PredictionResult(BaseModel):
    id: str
    symbol: str
    horizon: str
    model_id: str
    model_name: str
    family: str
    branch: str
    signal: SignalType
    confidence: float
    predicted_move_pct: float
    forecast_horizon: str
    last_price: float
    reference_price: float
    generated_at: str
    reasoning: str
    execution_mode: str
    forecast_path: list[ForecastPoint] = Field(default_factory=list)
    auto_run: bool = False
    lookback: int | None = None
    pred_len: int | None = None


class PredictionLogEnvelope(BaseModel):
    items: list[PredictionResult]


class SystemSummary(BaseModel):
    live_status: LiveStatus
    symbols: list[str]
    chart_intervals: list[str]
    forecast_horizons: list[str]
    model_count: int
    latest_predictions: int


class WsEnvelope(BaseModel):
    type: str
    payload: dict
