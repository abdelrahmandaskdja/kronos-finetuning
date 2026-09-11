from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import Candle


FEATURE_NAMES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_return",
    "body_pct",
    "range_pct",
)


def candles_to_feature_matrix(candles: Sequence[Candle]) -> np.ndarray:
    if not candles:
        raise ValueError("Cannot preprocess an empty candle window")

    rows: list[list[float]] = []
    previous_close = float(candles[0].close)
    for candle in candles:
        open_price = float(candle.open)
        high_price = float(candle.high)
        low_price = float(candle.low)
        close_price = float(candle.close)
        volume = float(candle.volume)
        close_return = 0.0 if previous_close == 0 else (close_price - previous_close) / previous_close
        body_pct = 0.0 if open_price == 0 else (close_price - open_price) / open_price
        range_pct = 0.0 if open_price == 0 else (high_price - low_price) / open_price
        rows.append(
            [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                close_return,
                body_pct,
                range_pct,
            ]
        )
        previous_close = close_price
    return np.asarray(rows, dtype=np.float32)


def build_model_input(candles: Sequence[Candle]) -> dict:
    features = candles_to_feature_matrix(candles)
    closes = features[:, 3]
    timestamps = np.asarray([candle.close_time for candle in candles], dtype=np.int64)
    return {
        "features": features,
        "feature_names": FEATURE_NAMES,
        "closes": closes,
        "timestamps": timestamps,
        "lookback": len(candles),
    }
