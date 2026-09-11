from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str
    api_prefix: str
    host: str
    port: int
    cors_origins: list[str]
    enable_binance_rest: bool
    use_mock_predictions: bool
    symbols: list[str]
    chart_intervals: list[str]
    forecast_horizons: list[str]
    bootstrap_candle_limit: int
    max_candle_limit: int
    binance_snapshot_poll_seconds: float
    binance_candle_poll_seconds: float
    stream_status_ttl_seconds: int
    database_path: Path
    binance_rest_url: str


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[3]
    backend_root = Path(__file__).resolve().parents[1]
    storage_dir = backend_root / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=os.getenv("MARKETISER_APP_NAME", "Marketiser API"),
        api_prefix=os.getenv("MARKETISER_API_PREFIX", "/api"),
        host=os.getenv("MARKETISER_HOST", "0.0.0.0"),
        port=int(os.getenv("MARKETISER_PORT", "8000")),
        cors_origins=_split_csv(
            os.getenv("MARKETISER_CORS_ORIGINS"),
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        ),
        enable_binance_rest=os.getenv(
            "MARKETISER_ENABLE_BINANCE_REST",
            os.getenv("MARKETISER_ENABLE_BINANCE_STREAM", "true"),
        ).lower()
        == "true",
        use_mock_predictions=os.getenv("MARKETISER_USE_MOCK_PREDICTIONS", "true").lower() == "true",
        symbols=_split_csv(
            os.getenv("MARKETISER_SYMBOLS"),
            ["BTCUSDT"],
        ),
        chart_intervals=_split_csv(
            os.getenv("MARKETISER_CHART_INTERVALS"),
            ["1m", "5m", "15m", "1h", "4h", "1d"],
        ),
        forecast_horizons=_split_csv(
            os.getenv("MARKETISER_FORECAST_HORIZONS"),
            ["5m", "1h", "1d"],
        ),
        bootstrap_candle_limit=int(os.getenv("MARKETISER_BOOTSTRAP_CANDLE_LIMIT", "180")),
        max_candle_limit=int(os.getenv("MARKETISER_MAX_CANDLE_LIMIT", "240")),
        binance_snapshot_poll_seconds=float(os.getenv("MARKETISER_BINANCE_SNAPSHOT_POLL_SECONDS", "5")),
        binance_candle_poll_seconds=float(os.getenv("MARKETISER_BINANCE_CANDLE_POLL_SECONDS", "20")),
        stream_status_ttl_seconds=int(os.getenv("MARKETISER_STREAM_STATUS_TTL", "20")),
        database_path=Path(os.getenv("MARKETISER_DATABASE_PATH", str(storage_dir / "marketiser.db"))),
        binance_rest_url=os.getenv("MARKETISER_BINANCE_REST_URL", "https://api.binance.com"),
    )


SETTINGS = load_settings()
REPO_ROOT = Path(__file__).resolve().parents[3]
Kronos_ROOT = REPO_ROOT
