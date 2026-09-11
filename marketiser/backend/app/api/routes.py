from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import PredictionLogEnvelope, PredictionRequest, SystemSummary

router = APIRouter()

TV_EXCHANGE = "BINANCE"


def _tv_symbol_base(symbol: str) -> str:
    return symbol.split(":")[-1].upper()


def _tv_resolution_to_interval(resolution: str) -> str:
    mapping = {
        "1": "1m",
        "5": "5m",
        "15": "15m",
        "60": "1h",
        "240": "4h",
        "D": "1d",
        "1D": "1d",
    }
    return mapping.get(resolution.upper(), "1h")


def _interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    return mapping.get(interval, 3600)


def _symbol_description(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]} / USDT"
    return symbol


def _symbol_pricescale(symbol: str) -> int:
    if symbol.startswith(("BTC", "ETH")):
        return 100
    if symbol.startswith("SOL"):
        return 1000
    return 10000


def _tv_symbol_info(symbol: str) -> dict:
    base = _tv_symbol_base(symbol)
    return {
        "ticker": f"{TV_EXCHANGE}:{base}",
        "name": base,
        "description": _symbol_description(base),
        "type": "crypto",
        "session": "24x7",
        "timezone": "Etc/UTC",
        "exchange": TV_EXCHANGE,
        "listed_exchange": TV_EXCHANGE,
        "format": "price",
        "minmov": 1,
        "pricescale": _symbol_pricescale(base),
        "has_intraday": True,
        "has_daily": True,
        "has_weekly_and_monthly": True,
        "has_empty_bars": False,
        "intraday_multipliers": ["1", "5", "15", "60", "240"],
        "daily_multipliers": ["1"],
        "supported_resolutions": ["1", "5", "15", "60", "240", "1D"],
        "volume_precision": 3,
        "data_status": "streaming",
        "visible_plots_set": "ohlcv",
    }


@router.get("/health")
async def health(request: Request) -> dict:
    registry = request.app.state.model_registry
    market_data = request.app.state.market_data
    return {
        "ok": True,
        "service": request.app.state.settings.app_name,
        "live_state": market_data.get_status().state,
        "models_loaded": registry.count(),
        "ws_clients": request.app.state.ws_manager.connection_count,
    }


@router.get("/system/summary", response_model=SystemSummary)
async def system_summary(request: Request) -> SystemSummary:
    market_data = request.app.state.market_data
    prediction_engine = request.app.state.prediction_engine
    settings = request.app.state.settings
    latest_logs = prediction_engine.list_logs(limit=12, symbols=settings.symbols)
    return SystemSummary(
        live_status=market_data.get_status(),
        symbols=settings.symbols,
        chart_intervals=settings.chart_intervals,
        forecast_horizons=settings.forecast_horizons,
        model_count=request.app.state.model_registry.count(),
        latest_predictions=len(latest_logs),
    )


@router.get("/watchlist")
async def watchlist(request: Request) -> dict:
    market_data = request.app.state.market_data
    return {"items": [item.model_dump() for item in market_data.get_watchlist()]}


@router.get("/market/candles")
async def market_candles(
    request: Request,
    symbol: str = Query(...),
    interval: str = Query(...),
    limit: int = Query(120, ge=10, le=240),
) -> dict:
    market_data = request.app.state.market_data
    return {"items": [item.model_dump() for item in market_data.get_candles(symbol, interval, limit=limit)]}


@router.get("/market/history")
async def market_history(
    request: Request,
    symbol: str = Query(...),
    interval: str = Query(...),
    from_ts: int = Query(..., alias="from", ge=0),
    to_ts: int = Query(..., alias="to", ge=0),
) -> dict:
    market_data = request.app.state.market_data
    items = await asyncio.to_thread(
        market_data.get_historical_candles,
        symbol,
        interval,
        from_seconds=from_ts,
        to_seconds=to_ts,
    )
    return {
        "items": [item.model_dump() for item in items]
    }


@router.get("/market/status")
async def market_status(request: Request) -> dict:
    return request.app.state.market_data.get_status().model_dump()


@router.get("/tv/config")
async def tv_config() -> dict:
    return {
        "supported_resolutions": ["1", "5", "15", "60", "240", "1D"],
        "supports_group_request": False,
        "supports_search": True,
        "supports_marks": False,
        "supports_timescale_marks": False,
        "supports_time": True,
        "exchanges": [{"value": TV_EXCHANGE, "name": TV_EXCHANGE, "desc": "Binance crypto exchange"}],
        "symbols_types": [{"name": "crypto", "value": "crypto"}],
    }


@router.get("/tv/search")
async def tv_search(
    request: Request,
    query: str = Query(default=""),
    exchange: str = Query(default=""),
    type: str = Query(default=""),
    limit: int = Query(default=30, ge=1, le=100),
) -> list[dict]:
    del type
    settings = request.app.state.settings
    exchange_filter = exchange.strip().upper()
    query_upper = query.strip().upper()
    results: list[dict] = []
    for symbol in settings.symbols:
        base = _tv_symbol_base(symbol)
        if exchange_filter and exchange_filter != TV_EXCHANGE:
            continue
        if query_upper and query_upper not in base and query_upper not in f"{TV_EXCHANGE}:{base}":
            continue
        results.append(
            {
                "symbol": base,
                "full_name": f"{TV_EXCHANGE}:{base}",
                "description": _symbol_description(base),
                "exchange": TV_EXCHANGE,
                "ticker": f"{TV_EXCHANGE}:{base}",
                "type": "crypto",
            }
        )
    return results[:limit]


@router.get("/tv/symbols")
async def tv_symbols(request: Request, symbol: str = Query(...)) -> dict:
    settings = request.app.state.settings
    base = _tv_symbol_base(symbol)
    if base not in {item.upper() for item in settings.symbols}:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")
    return _tv_symbol_info(base)


@router.get("/tv/history")
async def tv_history(
    request: Request,
    symbol: str = Query(...),
    resolution: str = Query(...),
    from_ts: int = Query(default=0, alias="from", ge=0),
    to_ts: int = Query(default=0, alias="to", ge=0),
    countback: int | None = Query(default=None, ge=1),
) -> dict:
    market_data = request.app.state.market_data
    base = _tv_symbol_base(symbol)
    interval = _tv_resolution_to_interval(resolution)
    interval_seconds = _interval_to_seconds(interval)
    safe_to = to_ts or int(datetime.now(tz=UTC).timestamp())
    safe_from = from_ts
    if countback is not None:
        safe_from = max(safe_to - (countback * interval_seconds), 0)
    if safe_to <= safe_from:
        safe_to = safe_from + interval_seconds

    items = await asyncio.to_thread(
        market_data.get_historical_candles,
        base,
        interval,
        from_seconds=safe_from,
        to_seconds=safe_to,
    )
    if not items:
        return {"s": "no_data"}
    return {
        "s": "ok",
        "t": [int(datetime.fromisoformat(item.open_time).timestamp()) for item in items],
        "o": [item.open for item in items],
        "h": [item.high for item in items],
        "l": [item.low for item in items],
        "c": [item.close for item in items],
        "v": [item.volume for item in items],
    }


@router.get("/tv/time")
async def tv_time() -> int:
    return int(datetime.now(tz=UTC).timestamp())


@router.get("/models")
async def list_models(
    request: Request,
    horizon: str | None = Query(default=None),
) -> dict:
    registry = request.app.state.model_registry
    return {"items": [model.model_dump() for model in registry.list_models(horizon=horizon)]}


@router.get("/model-families")
async def model_families(request: Request) -> dict:
    return {"items": [family.model_dump() for family in request.app.state.model_registry.families()]}


@router.post("/predictions/run")
async def run_prediction(request: Request, payload: PredictionRequest) -> dict:
    prediction_engine = request.app.state.prediction_engine
    registry = request.app.state.model_registry
    try:
        registry.get_model(payload.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {payload.model_id}") from exc
    try:
        result = await prediction_engine.run_prediction(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@router.get("/predictions/latest")
async def latest_prediction(
    request: Request,
    symbol: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
) -> dict:
    prediction_engine = request.app.state.prediction_engine
    latest = prediction_engine.latest_prediction(symbol=symbol, horizon=horizon, model_id=model_id)
    return {"item": latest.model_dump() if latest else None}


@router.get("/predictions/logs", response_model=PredictionLogEnvelope)
async def prediction_logs(request: Request, limit: int = Query(20, ge=1, le=100)) -> PredictionLogEnvelope:
    items = request.app.state.prediction_engine.list_logs(limit=limit, symbols=request.app.state.settings.symbols)
    return PredictionLogEnvelope(items=items)
