from __future__ import annotations

import asyncio
import math
import statistics
import sys
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Kronos_ROOT
from ..db import PredictionStore
from ..schemas import CandlePoint, ForecastPoint, ModelRecord, PredictionRequest, PredictionResult
from ..websocket_manager import WebSocketManager
from .market_data import MarketDataService
from .model_registry import ModelRegistry

if str(Kronos_ROOT) not in sys.path:
    sys.path.append(str(Kronos_ROOT))

try:
    import torch
    from finetune_csv.eval_direction_model import load_checkpoint_model
    from model import Kronos, KronosPredictor, KronosTokenizer

    KRONOS_RUNTIME_AVAILABLE = True
except ImportError:
    Kronos = None
    KronosPredictor = None
    KronosTokenizer = None
    load_checkpoint_model = None
    torch = None
    KRONOS_RUNTIME_AVAILABLE = False


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def horizon_delta(horizon: str) -> timedelta:
    mapping = {
        "5m": timedelta(minutes=5),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }
    return mapping.get(horizon, timedelta(minutes=5))


class PredictionEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        market_data: MarketDataService,
        store: PredictionStore,
        ws_manager: WebSocketManager,
    ) -> None:
        self.registry = registry
        self.market_data = market_data
        self.store = store
        self.ws_manager = ws_manager
        self._predictors: dict[str, KronosPredictor] = {}
        self._direction_bundles: dict[str, dict[str, Any]] = {}
        self._predictor_lock = threading.Lock()

    async def run_prediction(self, request: PredictionRequest) -> PredictionResult:
        model = self.registry.get_model(request.model_id)
        if not model.supports_live_inference:
            raise ValueError(f"{model.display_name} is not configured for live inference.")

        runner = self._run_direction_head_prediction if model.execution_mode == "kronos-direction-head" else self._run_kronos_prediction
        result = await asyncio.to_thread(runner, request, model)
        self.store.log_prediction(result)
        await self.ws_manager.broadcast({"type": "prediction.created", "payload": result.model_dump()})
        return result

    def list_logs(self, limit: int = 30, symbols: list[str] | None = None) -> list[PredictionResult]:
        items = self.store.list_predictions(limit=max(limit * 4, limit))
        if symbols:
            symbol_set = set(symbols)
            items = [item for item in items if item.symbol in symbol_set]
        return items[:limit]

    def latest_prediction(
        self,
        symbol: str | None = None,
        horizon: str | None = None,
        model_id: str | None = None,
    ) -> PredictionResult | None:
        return self.store.latest_prediction(symbol=symbol, horizon=horizon, model_id=model_id)

    def _run_kronos_prediction(self, request: PredictionRequest, model: ModelRecord) -> PredictionResult:
        if not KRONOS_RUNTIME_AVAILABLE:
            raise ValueError("Kronos runtime is unavailable in this environment.")
        if not model.runtime_model_id or not model.runtime_tokenizer_id:
            raise ValueError(f"{model.display_name} does not define a runtime model source.")

        lookback = self._resolve_lookback(request, model)
        pred_len = self._resolve_pred_len(request, model)
        candles = self._load_candles(request.symbol, request.horizon, lookback)
        predictor = self._get_predictor(model)

        context_frame = pd.DataFrame(
            {
                "open": [candle.open for candle in candles],
                "high": [candle.high for candle in candles],
                "low": [candle.low for candle in candles],
                "close": [candle.close for candle in candles],
                "volume": [candle.volume for candle in candles],
            }
        )
        x_timestamp = pd.Series(pd.to_datetime([candle.open_time for candle in candles]), name="timestamps")
        next_times = pd.date_range(
            start=pd.Timestamp(candles[-1].open_time) + horizon_delta(request.horizon),
            periods=pred_len,
            freq=horizon_delta(request.horizon),
        )
        y_timestamp = pd.Series(next_times, name="timestamps")

        prediction_frame = predictor.predict(
            df=context_frame,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=0.95,
            sample_count=1,
            verbose=False,
        )

        last_price = candles[-1].close
        recent_returns = [pct_change(current.close, previous.close) for previous, current in zip(candles[:-1], candles[1:])]
        realized_vol_pct = stdev(recent_returns[-min(len(recent_returns), 32) :]) * 100 if recent_returns else 0.0
        forecast_path = self._forecast_from_frame(
            prediction_frame,
            reference_price=last_price,
            horizon=request.horizon,
            realized_vol_pct=realized_vol_pct,
            stabilize=model.scope == "finetuned",
        )
        reference_price = candles[-2].close if len(candles) > 1 else candles[-1].close
        final_price = forecast_path[-1].price if forecast_path else last_price
        predicted_move_pct = ((final_price / last_price) - 1) * 100 if last_price else 0.0
        neutral_band_pct = max(realized_vol_pct * 0.35, 0.12)
        if predicted_move_pct > neutral_band_pct:
            signal = "bullish"
        elif predicted_move_pct < -neutral_band_pct:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = min(max(0.26 + abs(predicted_move_pct) / max(realized_vol_pct * 1.9, 0.45), 0.24), 0.97)
        if signal == "neutral":
            confidence = min(confidence, 0.58)

        reasoning = self._build_reasoning(
            model=model,
            horizon=request.horizon,
            lookback=lookback,
            pred_len=pred_len,
            predicted_move_pct=predicted_move_pct,
            realized_vol_pct=realized_vol_pct,
            last_price=last_price,
            final_price=final_price,
            signal=signal,
        )

        return PredictionResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            horizon=request.horizon,
            model_id=request.model_id,
            model_name=model.display_name,
            family=model.family,
            branch=model.branch,
            signal=signal,
            confidence=round(confidence, 4),
            predicted_move_pct=round(predicted_move_pct, 4),
            forecast_horizon=request.horizon,
            last_price=round(last_price, 6),
            reference_price=round(reference_price, 6),
            generated_at=utc_now_iso(),
            reasoning=reasoning,
            execution_mode=model.execution_mode,
            forecast_path=forecast_path,
            auto_run=request.auto_run,
            lookback=lookback,
            pred_len=pred_len,
        )

    def _run_direction_head_prediction(self, request: PredictionRequest, model: ModelRecord) -> PredictionResult:
        if not KRONOS_RUNTIME_AVAILABLE or load_checkpoint_model is None or torch is None:
            raise ValueError("Direction-head runtime is unavailable in this environment.")
        if not model.checkpoint_path:
            raise ValueError(f"{model.display_name} does not define a direction-head checkpoint.")

        lookback = self._resolve_lookback(request, model)
        pred_len = self._resolve_pred_len(request, model)
        candles = self._load_candles(request.symbol, request.horizon, lookback)
        bundle = self._get_direction_bundle(model)

        feature_frame = np.asarray(
            [[candle.open, candle.high, candle.low, candle.close, candle.volume, candle.close * candle.volume] for candle in candles],
            dtype=np.float32,
        )
        clip = float(bundle["config"].get("clip", 5.0))
        feature_mean = feature_frame.mean(axis=0)
        feature_std = feature_frame.std(axis=0)
        feature_frame = np.clip((feature_frame - feature_mean) / (feature_std + 1e-5), -clip, clip).astype(np.float32)

        timestamps = pd.to_datetime([candle.open_time for candle in candles], utc=True)
        time_frame = np.asarray(
            [[stamp.minute, stamp.hour, stamp.weekday(), stamp.day, stamp.month] for stamp in timestamps],
            dtype=np.float32,
        )

        x_tensor = torch.from_numpy(feature_frame[None, :, :]).to(bundle["device"])
        stamp_tensor = torch.from_numpy(time_frame[None, :, :]).to(bundle["device"])
        with torch.no_grad():
            token_s1, token_s2 = bundle["tokenizer"].encode(x_tensor, half=True)
            logits = bundle["model"](token_s1, token_s2, stamp_tensor)
            prob_up = float(torch.sigmoid(logits).detach().cpu().reshape(-1)[0].item())

        last_price = candles[-1].close
        reference_price = candles[-2].close if len(candles) > 1 else last_price
        recent_returns = [pct_change(current.close, previous.close) for previous, current in zip(candles[:-1], candles[1:])]
        recent_abs_returns_pct = [abs(value) * 100 for value in recent_returns]
        realized_vol_pct = stdev(recent_returns[-min(len(recent_returns), 32) :]) * 100 if recent_returns else 0.0
        average_abs_move_pct = (
            statistics.fmean(recent_abs_returns_pct[-min(len(recent_abs_returns_pct), 32) :])
            if recent_abs_returns_pct
            else 0.0
        )

        forecast_path = self._project_direction_head_forecast(
            candles=candles,
            horizon=request.horizon,
            pred_len=pred_len,
            prob_up=prob_up,
            realized_vol_pct=realized_vol_pct,
            average_abs_move_pct=average_abs_move_pct,
        )
        final_price = forecast_path[-1].price if forecast_path else last_price
        predicted_move_pct = ((final_price / last_price) - 1) * 100 if last_price else 0.0

        if prob_up >= 0.54:
            signal = "bullish"
        elif prob_up <= 0.46:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = min(max(0.2 + abs(prob_up - 0.5) * 1.9, 0.2), 0.97)
        if signal == "neutral":
            confidence = min(confidence, 0.58)

        reasoning = self._build_direction_reasoning(
            model=model,
            horizon=request.horizon,
            lookback=lookback,
            pred_len=pred_len,
            prob_up=prob_up,
            predicted_move_pct=predicted_move_pct,
            realized_vol_pct=realized_vol_pct,
            last_price=last_price,
            final_price=final_price,
            signal=signal,
        )

        return PredictionResult(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            horizon=request.horizon,
            model_id=request.model_id,
            model_name=model.display_name,
            family=model.family,
            branch=model.branch,
            signal=signal,
            confidence=round(confidence, 4),
            predicted_move_pct=round(predicted_move_pct, 4),
            forecast_horizon=request.horizon,
            last_price=round(last_price, 6),
            reference_price=round(reference_price, 6),
            generated_at=utc_now_iso(),
            reasoning=reasoning,
            execution_mode=model.execution_mode,
            forecast_path=forecast_path,
            auto_run=request.auto_run,
            lookback=lookback,
            pred_len=pred_len,
        )

    def _resolve_lookback(self, request: PredictionRequest, model: ModelRecord) -> int:
        max_context = model.context_length or 512
        default_lookback = model.default_lookback or min(256, max_context)
        requested = default_lookback if not model.supports_custom_lookback else request.lookback or default_lookback
        return max(16, min(int(requested), max_context))

    def _resolve_pred_len(self, request: PredictionRequest, model: ModelRecord) -> int:
        default_pred_len = model.default_pred_len or 16
        requested = default_pred_len if not model.supports_custom_pred_len else request.pred_len or default_pred_len
        return max(1, min(int(requested), 64))

    def _load_candles(self, symbol: str, horizon: str, lookback: int) -> list[CandlePoint]:
        limit = max(lookback + 6, lookback)
        candles = self.market_data.get_candles(symbol, horizon, limit=limit)
        closed_candles = [candle for candle in candles if candle.is_closed]
        selected = closed_candles if len(closed_candles) >= lookback else candles
        if len(selected) < lookback:
            raise ValueError(
                f"Not enough {horizon} candle history for lookback={lookback}. "
                f"Only {len(selected)} candles are currently available."
            )
        return selected[-lookback:]

    def _get_predictor(self, model: ModelRecord) -> KronosPredictor:
        existing = self._predictors.get(model.id)
        if existing is not None:
            return existing

        with self._predictor_lock:
            existing = self._predictors.get(model.id)
            if existing is not None:
                return existing

            tokenizer = KronosTokenizer.from_pretrained(model.runtime_tokenizer_id)
            runtime_model = Kronos.from_pretrained(model.runtime_model_id)
            predictor = KronosPredictor(
                runtime_model,
                tokenizer,
                device="cpu",
                max_context=model.context_length or 512,
                clip=5,
            )
            self._predictors[model.id] = predictor
            return predictor

    def _get_direction_bundle(self, model: ModelRecord) -> dict[str, Any]:
        existing = self._direction_bundles.get(model.id)
        if existing is not None:
            return existing

        with self._predictor_lock:
            existing = self._direction_bundles.get(model.id)
            if existing is not None:
                return existing

            resolved_device = torch.device("cpu")
            checkpoint_path = Path(model.checkpoint_path).expanduser().resolve()
            preview = torch.load(checkpoint_path, map_location=resolved_device)
            preview_config = preview.get("config", {})
            predictor_path = self._resolve_runtime_path(
                preview.get("predictor_path") or preview_config.get("pretrained_predictor_path")
            )
            ckpt, direction_model, _, _ = load_checkpoint_model(
                checkpoint_path=checkpoint_path,
                device=resolved_device,
                predictor_path_override=predictor_path,
            )
            config = ckpt.get("config", {})
            tokenizer_path = self._resolve_runtime_path(
                ckpt.get("tokenizer_path") or config.get("pretrained_tokenizer_path")
            )
            if not tokenizer_path:
                raise ValueError(f"Could not resolve tokenizer path for {model.display_name}.")

            tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
            tokenizer.to(resolved_device)
            tokenizer.eval()
            direction_model.eval()

            bundle = {
                "device": resolved_device,
                "config": config,
                "model": direction_model,
                "tokenizer": tokenizer,
            }
            self._direction_bundles[model.id] = bundle
            return bundle

    def _resolve_runtime_path(self, value: object) -> str | None:
        if value is None:
            return None
        raw = str(value)
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        repo_candidate = (Kronos_ROOT / candidate).resolve()
        if repo_candidate.exists():
            return str(repo_candidate)
        return raw

    def _project_direction_head_forecast(
        self,
        *,
        candles: list[CandlePoint],
        horizon: str,
        pred_len: int,
        prob_up: float,
        realized_vol_pct: float,
        average_abs_move_pct: float,
    ) -> list[ForecastPoint]:
        if not candles:
            return []

        floors = {
            "5m": 0.12,
            "1h": 0.28,
            "1d": 0.9,
        }
        step_cap_pct = self._step_cap_pct(horizon=horizon, realized_vol_pct=realized_vol_pct)
        move_scale_pct = max(average_abs_move_pct, realized_vol_pct, floors.get(horizon, 0.2))
        move_scale_pct = min(move_scale_pct, step_cap_pct * 0.75)
        recent_volumes = [candle.volume for candle in candles[-min(len(candles), 32) :]]
        baseline_volume = statistics.fmean(recent_volumes) if recent_volumes else candles[-1].volume
        time_delta = horizon_delta(horizon)
        last_timestamp = pd.Timestamp(candles[-1].open_time)
        previous_close = candles[-1].close
        direction_strength = max(min((prob_up - 0.5) * 2.4, 1.0), -1.0)

        points: list[ForecastPoint] = []
        for index in range(1, pred_len + 1):
            step_strength = direction_strength * (0.88 ** (index - 1))
            projected_move_pct = step_strength * move_scale_pct
            open_price = previous_close
            close_price = max(open_price * (1 + projected_move_pct / 100), 1e-9)
            wick_pct = min(max(move_scale_pct * (0.35 + abs(step_strength) * 0.25), 0.08), step_cap_pct * 0.4)
            upper_wick_pct = wick_pct * (1.0 if step_strength >= 0 else 0.45)
            lower_wick_pct = wick_pct * (1.0 if step_strength < 0 else 0.45)
            body_high = max(open_price, close_price)
            body_low = min(open_price, close_price)
            high_price = body_high * (1 + upper_wick_pct / 100)
            low_price = max(body_low * (1 - lower_wick_pct / 100), 1e-9)
            timestamp = (last_timestamp + (time_delta * index)).isoformat()
            volume = max(baseline_volume * (1 + abs(step_strength) * 0.18), 0.0)
            points.append(
                ForecastPoint(
                    index=index,
                    timestamp=timestamp,
                    open=round(open_price, 6),
                    high=round(high_price, 6),
                    low=round(low_price, 6),
                    close=round(close_price, 6),
                    volume=round(volume, 6),
                    price=round(close_price, 6),
                    pct_from_last=round(((close_price / candles[-1].close) - 1) * 100, 4) if candles[-1].close else 0.0,
                )
            )
            previous_close = close_price

        return points

    def _forecast_from_frame(
        self,
        prediction_frame: pd.DataFrame,
        *,
        reference_price: float,
        horizon: str,
        realized_vol_pct: float,
        stabilize: bool,
    ) -> list[ForecastPoint]:
        raw_rows = [
            self._extract_forecast_row(timestamp=timestamp, row=row, reference_price=reference_price)
            for timestamp, row in prediction_frame.iterrows()
        ]
        rows = (
            self._stabilize_forecast_rows(
                raw_rows,
                reference_price=reference_price,
                horizon=horizon,
                realized_vol_pct=realized_vol_pct,
            )
            if stabilize
            else self._sanitize_forecast_rows(raw_rows, reference_price=reference_price)
        )

        points: list[ForecastPoint] = []
        for index, row in enumerate(rows, start=1):
            close_price = row["close"]
            points.append(
                ForecastPoint(
                    index=index,
                    timestamp=row["timestamp"],
                    open=round(row["open"], 6),
                    high=round(row["high"], 6),
                    low=round(row["low"], 6),
                    close=round(close_price, 6),
                    volume=round(row["volume"], 6),
                    price=round(close_price, 6),
                    pct_from_last=round(((close_price / reference_price) - 1) * 100, 4) if reference_price else 0.0,
                )
            )
        return points

    def _extract_forecast_row(
        self,
        *,
        timestamp: pd.Timestamp,
        row: pd.Series,
        reference_price: float,
    ) -> dict[str, float | str]:
        open_price = self._finite_price(row.get("open"), reference_price)
        close_price = self._finite_price(row.get("close"), open_price)
        high_price = self._finite_price(row.get("high"), max(open_price, close_price))
        low_price = self._finite_price(row.get("low"), min(open_price, close_price))
        return {
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": self._finite_volume(row.get("volume")),
        }

    def _sanitize_forecast_rows(
        self,
        rows: list[dict[str, float | str]],
        *,
        reference_price: float,
    ) -> list[dict[str, float | str]]:
        sanitized: list[dict[str, float | str]] = []
        previous_close = reference_price

        for row in rows:
            open_price = self._finite_price(row.get("open"), previous_close)
            close_price = self._finite_price(row.get("close"), open_price)
            high_price = max(self._finite_price(row.get("high"), max(open_price, close_price)), open_price, close_price)
            low_candidate = self._finite_price(row.get("low"), min(open_price, close_price))
            low_price = min(low_candidate, open_price, close_price)
            sanitized.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "open": open_price,
                    "high": high_price,
                    "low": max(low_price, 1e-9),
                    "close": close_price,
                    "volume": self._finite_volume(row.get("volume")),
                }
            )
            previous_close = close_price

        return sanitized

    def _stabilize_forecast_rows(
        self,
        rows: list[dict[str, float | str]],
        *,
        reference_price: float,
        horizon: str,
        realized_vol_pct: float,
    ) -> list[dict[str, float | str]]:
        if not rows or reference_price <= 0:
            return rows

        step_cap_pct = self._step_cap_pct(horizon=horizon, realized_vol_pct=realized_vol_pct)
        wick_cap_pct = min(max(step_cap_pct * 0.45, 0.4), 6.0)
        stabilized: list[dict[str, float | str]] = []
        previous = reference_price

        for row in rows:
            if previous <= 0:
                previous = reference_price

            raw_close = self._finite_price(row.get("close"), previous)
            if not math.isfinite(raw_close) or raw_close <= 0:
                raw_move_pct = -step_cap_pct
            else:
                raw_move_pct = ((raw_close / previous) - 1) * 100
            bounded_move_pct = max(-step_cap_pct, min(step_cap_pct, raw_move_pct))
            stable_open = previous
            stable_close = previous * (1 + bounded_move_pct / 100)

            raw_open = self._finite_price(row.get("open"), previous)
            raw_high = self._finite_price(row.get("high"), max(raw_open, raw_close))
            raw_low = self._finite_price(row.get("low"), min(raw_open, raw_close))
            raw_body_high = max(raw_open, raw_close)
            raw_body_low = min(raw_open, raw_close)

            upper_wick_pct = 0.0
            if raw_body_high > 0:
                upper_wick_pct = max(((raw_high / raw_body_high) - 1) * 100, 0.0)

            lower_wick_pct = 0.0
            if raw_body_low > 0:
                lower_wick_pct = max((1 - (raw_low / raw_body_low)) * 100, 0.0)

            upper_wick_pct = min(upper_wick_pct, wick_cap_pct)
            lower_wick_pct = min(lower_wick_pct, wick_cap_pct)

            body_high = max(stable_open, stable_close)
            body_low = min(stable_open, stable_close)
            stable_high = body_high * (1 + upper_wick_pct / 100)
            stable_low = max(body_low * (1 - lower_wick_pct / 100), 1e-9)

            stabilized.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "open": stable_open,
                    "high": stable_high,
                    "low": stable_low,
                    "close": stable_close,
                    "volume": self._finite_volume(row.get("volume")),
                }
            )
            previous = stable_close

        return stabilized

    def _finite_price(self, value: object, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not math.isfinite(number) or number <= 0:
            return float(fallback)
        return number

    def _finite_volume(self, value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number) or number < 0:
            return 0.0
        return number

    def _step_cap_pct(self, *, horizon: str, realized_vol_pct: float) -> float:
        floors = {
            "5m": 1.2,
            "1h": 3.5,
            "1d": 6.0,
        }
        multipliers = {
            "5m": 6.0,
            "1h": 4.0,
            "1d": 3.0,
        }
        floor = floors.get(horizon, 2.0)
        multiplier = multipliers.get(horizon, 4.0)
        return min(max(realized_vol_pct * multiplier, floor), 18.0)

    def _build_reasoning(
        self,
        *,
        model: ModelRecord,
        horizon: str,
        lookback: int,
        pred_len: int,
        predicted_move_pct: float,
        realized_vol_pct: float,
        last_price: float,
        final_price: float,
        signal: str,
    ) -> str:
        stance = {
            "bullish": "Kronos forecasts upward continuation across the generated path",
            "bearish": "Kronos forecasts downside continuation across the generated path",
            "neutral": "Kronos sees a low-conviction path relative to recent volatility",
        }[signal]
        return (
            f"{stance}. Model={model.display_name}, horizon={horizon}, lookback={lookback}, pred_len={pred_len}. "
            f"Last close={last_price:.2f}, forecast end close={final_price:.2f}, move={predicted_move_pct:.2f}%, "
            f"recent realized volatility={realized_vol_pct:.2f}%."
        )

    def _build_direction_reasoning(
        self,
        *,
        model: ModelRecord,
        horizon: str,
        lookback: int,
        pred_len: int,
        prob_up: float,
        predicted_move_pct: float,
        realized_vol_pct: float,
        last_price: float,
        final_price: float,
        signal: str,
    ) -> str:
        stance = {
            "bullish": "The hourly direction head leans up on the next-bar classification",
            "bearish": "The hourly direction head leans down on the next-bar classification",
            "neutral": "The hourly direction head is near coin-flip on the next bar",
        }[signal]
        return (
            f"{stance}. Model={model.display_name}, horizon={horizon}, lookback={lookback}, pred_len={pred_len}. "
            f"Prob_up={prob_up:.4f}, last close={last_price:.2f}, projected end close={final_price:.2f}, "
            f"projected move={predicted_move_pct:.2f}%, recent realized volatility={realized_vol_pct:.2f}%. "
            "The displayed OHLCV path is a Marketiser projection built from the direction probability because this checkpoint is a direction-head model, not a native multi-bar OHLC forecaster."
        )
