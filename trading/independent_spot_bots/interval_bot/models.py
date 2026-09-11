from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import sys

import numpy as np
import pandas as pd

from .preprocessing import build_model_input
from .types import Candle, ModelPrediction, PredictionStep, SignalDirection, utc_now_ms

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_csv.eval_consecutive_rollout import build_stability_profile, load_csv as load_rollout_csv, stabilize_pred_block
from model import Kronos, KronosPredictor, KronosTokenizer


def _direction_from_value(value: float) -> SignalDirection:
    return SignalDirection.UP if value >= 0 else SignalDirection.DOWN


def _coerce_single_step(raw: Any) -> PredictionStep:
    if isinstance(raw, PredictionStep):
        return raw
    if isinstance(raw, dict):
        if "direction" in raw:
            direction = SignalDirection(str(raw["direction"]).upper())
            score = float(raw.get("score", 1.0 if direction == SignalDirection.UP else -1.0))
            return PredictionStep(
                direction=direction,
                score=score,
                raw_value=raw.get("raw_value"),
                label=raw.get("label"),
                metadata=dict(raw.get("metadata", {})),
            )
        if "signal" in raw:
            return _coerce_single_step(raw["signal"])
        if "score" in raw:
            score = float(raw["score"])
            return PredictionStep(direction=_direction_from_value(score), score=score, raw_value=score)
    if isinstance(raw, str):
        direction = SignalDirection(str(raw).upper())
        score = 1.0 if direction == SignalDirection.UP else -1.0
        return PredictionStep(direction=direction, score=score, raw_value=score)
    if isinstance(raw, (int, float, bool, np.number)):
        score = float(raw)
        return PredictionStep(direction=_direction_from_value(score), score=score, raw_value=score)
    raise TypeError(f"Unsupported prediction step type: {type(raw)!r}")


def _coerce_prediction(raw: Any, predlen: int, source: str) -> ModelPrediction:
    if isinstance(raw, ModelPrediction):
        return raw.validate(predlen)

    if isinstance(raw, dict) and "steps" in raw:
        steps = [_coerce_single_step(item) for item in raw["steps"]]
        return ModelPrediction(
            steps=steps,
            generated_at_ms=utc_now_ms(),
            source=source,
            metadata={key: value for key, value in raw.items() if key != "steps"},
        ).validate(predlen)

    if predlen == 1:
        return ModelPrediction(
            steps=[_coerce_single_step(raw)],
            generated_at_ms=utc_now_ms(),
            source=source,
        ).validate(predlen)

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        steps = [_coerce_single_step(item) for item in raw]
        return ModelPrediction(steps=steps, generated_at_ms=utc_now_ms(), source=source).validate(predlen)

    raise ValueError("Multi-step models must return a sequence or a dict with a 'steps' field")


class BaseModelAdapter(ABC):
    def __init__(self, model_path: Path, predlen: int, lookback: int, logger, python_callable: str | None = None) -> None:
        self.model_path = model_path
        self.predlen = predlen
        self.lookback = lookback
        self.logger = logger
        self.python_callable = python_callable

    def preprocess(self, candles: Sequence[Candle]) -> dict:
        if len(candles) < self.lookback:
            raise ValueError(f"Expected at least {self.lookback} candles, got {len(candles)}")
        return build_model_input(candles[-self.lookback :])

    @abstractmethod
    def predict(self, candles: Sequence[Candle]) -> ModelPrediction:
        raise NotImplementedError


class StubDirectionalAdapter(BaseModelAdapter):
    def predict(self, candles: Sequence[Candle]) -> ModelPrediction:
        payload = self.preprocess(candles)
        closes = payload["closes"]

        fast_window = closes[-min(len(closes), 8) :]
        slow_window = closes[-min(len(closes), 32) :]
        fast_avg = float(fast_window.mean())
        slow_avg = float(slow_window.mean())
        baseline_score = 0.0 if slow_avg == 0 else (fast_avg - slow_avg) / slow_avg

        anchor_index = max(0, len(closes) - min(len(closes), 5))
        anchor_close = float(closes[anchor_index])
        latest_close = float(closes[-1])
        momentum_score = 0.0 if anchor_close == 0 else (latest_close - anchor_close) / anchor_close

        steps: list[PredictionStep] = []
        for step_index in range(self.predlen):
            decay = max(0.25, 1.0 - (step_index / max(self.predlen, 1)))
            step_score = baseline_score + (momentum_score * decay)
            steps.append(
                PredictionStep(
                    direction=_direction_from_value(step_score),
                    score=float(step_score),
                    raw_value=float(step_score),
                    label=f"stub_step_{step_index}",
                    metadata={
                        "note": "TODO replace the stub model with your trained inference adapter",
                        "baseline_score": baseline_score,
                        "momentum_score": momentum_score,
                    },
                )
            )

        return ModelPrediction(
            steps=steps,
            generated_at_ms=utc_now_ms(),
            source="stub",
            metadata={"model_path": str(self.model_path)},
        ).validate(self.predlen)


class PythonFunctionAdapter(BaseModelAdapter):
    def __init__(self, model_path: Path, predlen: int, lookback: int, logger, python_callable: str | None = None) -> None:
        super().__init__(model_path, predlen, lookback, logger, python_callable)
        if not python_callable or ":" not in python_callable:
            raise ValueError("python_callable must be in the form 'module.submodule:function_name'")
        module_name, function_name = python_callable.split(":", 1)
        module = importlib.import_module(module_name)
        self._callable = getattr(module, function_name)

    def predict(self, candles: Sequence[Candle]) -> ModelPrediction:
        payload = self.preprocess(candles)
        raw = self._callable(
            payload,
            predlen=self.predlen,
            lookback=self.lookback,
            model_path=str(self.model_path),
        )
        return _coerce_prediction(raw, self.predlen, source=self.python_callable or "python_callable")


class _LoadedObjectAdapter(BaseModelAdapter):
    source_name = "loaded_object"

    def __init__(self, model_path: Path, predlen: int, lookback: int, logger, python_callable: str | None = None) -> None:
        super().__init__(model_path, predlen, lookback, logger, python_callable)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        self._model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        raise NotImplementedError

    def _invoke_loaded_model(self, payload: dict) -> Any:
        model = self._model
        features = payload["features"]

        if hasattr(model, "predict_directions"):
            return model.predict_directions(payload, predlen=self.predlen)
        if hasattr(model, "predict"):
            try:
                return model.predict(payload)
            except TypeError:
                try:
                    return model.predict(features[np.newaxis, ...], verbose=0)
                except TypeError:
                    return model.predict(features[np.newaxis, ...])
        if callable(model):
            try:
                return model(payload, predlen=self.predlen)
            except TypeError:
                return model(features[np.newaxis, ...])
        raise TypeError(
            "Loaded model object must expose predict_directions(), predict(), or be callable"
        )

    def predict(self, candles: Sequence[Candle]) -> ModelPrediction:
        payload = self.preprocess(candles)
        raw = self._invoke_loaded_model(payload)
        return _coerce_prediction(raw, self.predlen, source=self.source_name)


class JoblibModelAdapter(_LoadedObjectAdapter):
    source_name = "joblib"

    def _load_model(self) -> Any:
        import joblib

        return joblib.load(self.model_path)


class PyTorchModelAdapter(_LoadedObjectAdapter):
    source_name = "pytorch"

    def _load_model(self) -> Any:
        import torch

        return torch.load(self.model_path, map_location="cpu")


class TensorFlowModelAdapter(_LoadedObjectAdapter):
    source_name = "tensorflow"

    def _load_model(self) -> Any:
        import tensorflow as tf

        return tf.keras.models.load_model(self.model_path)


def infer_time_delta(timestamps: pd.Series) -> pd.Timedelta | None:
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    try:
        return diffs.mode().iloc[0]
    except (IndexError, ValueError):
        return diffs.iloc[-1]


def build_prediction_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        columns.append("volume")
    if "amount" in df.columns:
        columns.append("amount")
    return df[columns].copy()


def sanitize_forecast_frame(df: pd.DataFrame, allow_non_positive: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    if not allow_non_positive and (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Forecast produced non-positive prices")
    return out


def apply_consecutive_direction_rule(direction_mode: str, forecast_df: pd.DataFrame, last_close: float) -> pd.DataFrame:
    out = forecast_df.copy()
    if direction_mode == "close_to_close":
        reference_close = float(last_close)
        ref_values: list[float] = []
        pred_dirs: list[int] = []
        move_pcts: list[float | None] = []
        for _, row in out.iterrows():
            ref_values.append(reference_close)
            pred_close = float(row["close"])
            pred_dirs.append(1 if pred_close >= reference_close else 0)
            if reference_close == 0:
                move_pcts.append(None)
            else:
                move_pcts.append(((pred_close / reference_close) - 1.0) * 100.0)
            reference_close = pred_close
        out["reference_close"] = ref_values
        out["pred_dir"] = pred_dirs
        out["move_pct"] = move_pcts
        return out

    base_open = out["open"].replace(0, np.nan)
    out["pred_dir"] = (out["close"] >= out["open"]).astype(int)
    out["move_pct"] = ((out["close"] / base_open) - 1.0) * 100.0
    return out


@lru_cache(maxsize=8)
def load_kronos_predictor(model_path: str, tokenizer_path: str, max_context: int, clip: float, device: str) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context, clip=clip)


@lru_cache(maxsize=8)
def load_stability_profile(context_path: str):
    args = SimpleNamespace(
        stability_quantile=0.995,
        stability_multiplier=6.0,
        min_return_cap=0.02,
        max_return_cap=1.0,
    )
    train_df = load_rollout_csv(Path(context_path).expanduser().resolve())
    return build_stability_profile(train_df, args)


def candles_to_market_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    rows = []
    for candle in candles:
        amount = float(candle.quote_asset_volume)
        if amount <= 0.0:
            amount = float(candle.close) * float(candle.volume)
        rows.append(
            {
                "timestamp": pd.to_datetime(int(candle.close_time), unit="ms"),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
                "amount": amount,
            }
        )
    return pd.DataFrame(rows)


class KronosConsecutiveAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_path: Path,
        predlen: int,
        lookback: int,
        logger,
        python_callable: str | None = None,
        *,
        tokenizer_path: Path,
        device: str,
        clip: float,
        direction_mode: str,
        stabilize_output: bool,
        stability_context_path: Path | None,
    ) -> None:
        super().__init__(model_path, predlen, lookback, logger, python_callable)
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.clip = clip
        self.direction_mode = direction_mode
        self.stabilize_output = stabilize_output
        self.stability_context_path = stability_context_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {self.tokenizer_path}")

    def _stabilize_forecast(self, context_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if not self.stabilize_output or self.stability_context_path is None:
            return forecast_df
        profile = load_stability_profile(str(self.stability_context_path))
        pred_block = forecast_df.rename(columns={"timestamp": "timestamps"})[
            ["timestamps", "open", "high", "low", "close", "volume", "amount"]
        ].copy()
        stable_block, _ = stabilize_pred_block(
            pred_block=pred_block,
            prev_close_seed=float(context_df["close"].iloc[-1]),
            profile=profile,
        )
        return sanitize_forecast_frame(
            stable_block.rename(columns={"timestamps": "timestamp"}),
            allow_non_positive=True,
        )

    def predict(self, candles: Sequence[Candle]) -> ModelPrediction:
        if len(candles) < self.lookback:
            raise ValueError(f"Expected at least {self.lookback} candles, got {len(candles)}")

        context_df = candles_to_market_frame(candles[-self.lookback :])
        time_delta = infer_time_delta(context_df["timestamp"])
        if time_delta is None:
            raise ValueError("Could not infer candle spacing for consecutive forecast")

        y_timestamp = pd.Series(
            pd.date_range(
                start=context_df["timestamp"].iloc[-1] + time_delta,
                periods=self.predlen,
                freq=time_delta,
            )
        )
        predictor = load_kronos_predictor(
            model_path=str(self.model_path),
            tokenizer_path=str(self.tokenizer_path),
            max_context=self.lookback,
            clip=self.clip,
            device=self.device,
        )
        pred_df = predictor.predict(
            build_prediction_input_frame(context_df),
            context_df["timestamp"],
            y_timestamp,
            pred_len=self.predlen,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )
        forecast_df = sanitize_forecast_frame(
            pred_df.reset_index().rename(columns={"index": "timestamp"}),
            allow_non_positive=True,
        )
        forecast_df = self._stabilize_forecast(context_df, forecast_df)
        forecast_df = apply_consecutive_direction_rule(
            self.direction_mode,
            forecast_df,
            float(context_df["close"].iloc[-1]),
        )

        steps: list[PredictionStep] = []
        for _, row in forecast_df.iterrows():
            move_pct = None if pd.isna(row.get("move_pct")) else float(row["move_pct"])
            direction = SignalDirection.UP if int(row["pred_dir"]) == 1 else SignalDirection.DOWN
            score = (move_pct / 100.0) if move_pct is not None else (1.0 if direction == SignalDirection.UP else -1.0)
            steps.append(
                PredictionStep(
                    direction=direction,
                    score=float(score),
                    raw_value=float(row["close"]),
                    label=pd.Timestamp(row["timestamp"]).isoformat(),
                    metadata={
                        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                        "reference_close": float(row.get("reference_close", np.nan)),
                        "predicted_close": float(row["close"]),
                        "move_pct": move_pct,
                        "direction_mode": self.direction_mode,
                    },
                )
            )

        return ModelPrediction(
            steps=steps,
            generated_at_ms=utc_now_ms(),
            source="kronos_consecutive",
            metadata={
                "model_path": str(self.model_path),
                "tokenizer_path": str(self.tokenizer_path),
                "direction_mode": self.direction_mode,
                "stabilize_output": self.stabilize_output,
                "stability_context_path": str(self.stability_context_path) if self.stability_context_path else "",
            },
        ).validate(self.predlen)


def build_model_adapter(
    model_type: str,
    model_path: Path,
    predlen: int,
    lookback: int,
    logger,
    python_callable: str | None = None,
    *,
    tokenizer_path: Path | None = None,
    device: str = "cpu",
    clip: float = 5.0,
    direction_mode: str = "close_to_close",
    stabilize_output: bool = True,
    stability_context_path: Path | None = None,
) -> BaseModelAdapter:
    normalized = model_type.strip().lower()
    if normalized == "stub":
        return StubDirectionalAdapter(model_path, predlen, lookback, logger, python_callable)
    if normalized == "kronos_consecutive":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required for kronos_consecutive models")
        return KronosConsecutiveAdapter(
            model_path,
            predlen,
            lookback,
            logger,
            python_callable,
            tokenizer_path=tokenizer_path,
            device=device,
            clip=clip,
            direction_mode=direction_mode,
            stabilize_output=stabilize_output,
            stability_context_path=stability_context_path,
        )
    if not model_path.exists():
        if normalized != "stub" and not model_path.exists():
            logger.warning("Model path %s does not exist. Falling back to stub adapter.", model_path)
        return StubDirectionalAdapter(model_path, predlen, lookback, logger, python_callable)
    if normalized == "python_function":
        return PythonFunctionAdapter(model_path, predlen, lookback, logger, python_callable)
    if normalized == "joblib":
        return JoblibModelAdapter(model_path, predlen, lookback, logger, python_callable)
    if normalized == "pytorch":
        return PyTorchModelAdapter(model_path, predlen, lookback, logger, python_callable)
    if normalized == "tensorflow":
        return TensorFlowModelAdapter(model_path, predlen, lookback, logger, python_callable)
    raise ValueError(f"Unsupported model_type: {model_type}")
