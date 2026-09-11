from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from .types import BalanceSnapshot, ModelPrediction, PredictionStep, Regime, SignalDirection, utc_now_ms


@dataclass(slots=True)
class BotState:
    last_processed_candle_close_time: int | None = None
    regime: Regime = Regime.CASH
    open_order_ids: list[int] = field(default_factory=list)
    last_balances: BalanceSnapshot = field(default_factory=BalanceSnapshot)
    forecast_steps: list[PredictionStep] = field(default_factory=list)
    forecast_index: int = 0
    forecast_cycle_started_at: int | None = None
    updated_at_ms: int = 0

    def has_active_forecast(self) -> bool:
        return bool(self.forecast_steps) and self.forecast_index < len(self.forecast_steps)

    def replace_forecast(self, prediction: ModelPrediction, cycle_started_at: int) -> None:
        self.forecast_steps = prediction.steps
        self.forecast_index = 0
        self.forecast_cycle_started_at = cycle_started_at
        self.updated_at_ms = utc_now_ms()

    def consume_next_forecast_step(self) -> PredictionStep:
        if not self.has_active_forecast():
            raise RuntimeError("No forecast step available to consume")
        step = self.forecast_steps[self.forecast_index]
        self.forecast_index += 1
        self.updated_at_ms = utc_now_ms()
        return step

    def clear_open_orders(self) -> None:
        self.open_order_ids = []
        self.updated_at_ms = utc_now_ms()

    def update_balances(self, balances: BalanceSnapshot) -> None:
        self.last_balances = balances
        self.updated_at_ms = utc_now_ms()

    def to_dict(self) -> dict:
        return {
            "last_processed_candle_close_time": self.last_processed_candle_close_time,
            "regime": self.regime.value,
            "open_order_ids": self.open_order_ids,
            "last_balances": {
                "base_free": str(self.last_balances.base_free),
                "base_locked": str(self.last_balances.base_locked),
                "quote_free": str(self.last_balances.quote_free),
                "quote_locked": str(self.last_balances.quote_locked),
                "captured_at_ms": self.last_balances.captured_at_ms,
            },
            "forecast_steps": [
                {
                    "direction": step.direction.value,
                    "score": step.score,
                    "raw_value": step.raw_value,
                    "label": step.label,
                    "metadata": step.metadata,
                }
                for step in self.forecast_steps
            ],
            "forecast_index": self.forecast_index,
            "forecast_cycle_started_at": self.forecast_cycle_started_at,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "BotState":
        if not payload:
            return cls()
        balances = payload.get("last_balances", {})
        forecast_steps = [
            PredictionStep(
                direction=SignalDirection(item["direction"]),
                score=float(item["score"]),
                raw_value=item.get("raw_value"),
                label=item.get("label"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("forecast_steps", [])
        ]
        return cls(
            last_processed_candle_close_time=payload.get("last_processed_candle_close_time"),
            regime=Regime(payload.get("regime", Regime.CASH.value)),
            open_order_ids=[int(value) for value in payload.get("open_order_ids", [])],
            last_balances=BalanceSnapshot(
                base_free=Decimal(str(balances.get("base_free", "0"))),
                base_locked=Decimal(str(balances.get("base_locked", "0"))),
                quote_free=Decimal(str(balances.get("quote_free", "0"))),
                quote_locked=Decimal(str(balances.get("quote_locked", "0"))),
                captured_at_ms=int(balances.get("captured_at_ms", 0)),
            ),
            forecast_steps=forecast_steps,
            forecast_index=int(payload.get("forecast_index", 0)),
            forecast_cycle_started_at=payload.get("forecast_cycle_started_at"),
            updated_at_ms=int(payload.get("updated_at_ms", 0)),
        )


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return BotState.from_dict(payload)

    def save(self, state: BotState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)
