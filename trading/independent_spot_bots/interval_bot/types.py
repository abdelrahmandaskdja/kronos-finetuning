from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import threading
from typing import Any


_TIME_OFFSET_MS = 0
_TIME_OFFSET_LOCK = threading.Lock()


def decimal_from(value: Any) -> Decimal:
    return Decimal(str(value))


def set_time_offset_ms(offset_ms: int) -> None:
    global _TIME_OFFSET_MS
    with _TIME_OFFSET_LOCK:
        _TIME_OFFSET_MS = int(offset_ms)


def current_time_offset_ms() -> int:
    with _TIME_OFFSET_LOCK:
        return _TIME_OFFSET_MS


def utc_now_ms() -> int:
    import time

    return int(time.time() * 1000) + current_time_offset_ms()


class Regime(str, Enum):
    LONG = "LONG"
    CASH = "CASH"


class SignalDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class ExecutionEnvironment(str, Enum):
    TESTNET = "testnet"
    LIVE = "live"


class PriceReference(str, Enum):
    BOOK = "book"
    CLOSE_OFFSET = "close_offset"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class Candle:
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int
    quote_asset_volume: Decimal
    number_of_trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    is_closed: bool = True


@dataclass(slots=True)
class PredictionStep:
    direction: SignalDirection
    score: float
    raw_value: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelPrediction:
    steps: list[PredictionStep]
    generated_at_ms: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, predlen: int) -> "ModelPrediction":
        if len(self.steps) != predlen:
            raise ValueError(f"Model returned {len(self.steps)} steps but predlen={predlen}")
        return self


@dataclass(slots=True)
class BalanceSnapshot:
    base_free: Decimal = Decimal("0")
    base_locked: Decimal = Decimal("0")
    quote_free: Decimal = Decimal("0")
    quote_locked: Decimal = Decimal("0")
    captured_at_ms: int = 0

    @property
    def total_base(self) -> Decimal:
        return self.base_free + self.base_locked

    @property
    def total_quote(self) -> Decimal:
        return self.quote_free + self.quote_locked


@dataclass(slots=True)
class ManagedOrder:
    order_id: int
    client_order_id: str
    side: OrderSide
    status: str
    price: Decimal
    orig_qty: Decimal
    executed_qty: Decimal
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolFilters:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    max_notional: Decimal | None
    price_precision: int | None
    quantity_precision: int | None
