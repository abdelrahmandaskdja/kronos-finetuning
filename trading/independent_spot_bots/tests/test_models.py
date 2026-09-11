from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interval_bot.models import KronosConsecutiveAdapter
from interval_bot.types import Candle, SignalDirection


def build_candle(index: int, close_price: str) -> Candle:
    open_time = index * 86_400_000
    close_time = open_time + 86_399_999
    close_decimal = Decimal(close_price)
    return Candle(
        open_time=open_time,
        open=close_decimal,
        high=close_decimal,
        low=close_decimal,
        close=close_decimal,
        volume=Decimal("10"),
        close_time=close_time,
        quote_asset_volume=close_decimal * Decimal("10"),
        number_of_trades=1,
        taker_buy_base_volume=Decimal("0"),
        taker_buy_quote_volume=Decimal("0"),
        is_closed=True,
    )


class _FakePredictor:
    def predict(self, *_args, **_kwargs):
        timestamps = _args[2]
        return pd.DataFrame(
            {
                "open": [101.0, 99.0],
                "high": [101.0, 101.0],
                "low": [99.0, 98.0],
                "close": [101.0, 99.0],
                "volume": [1.0, 1.0],
                "amount": [101.0, 99.0],
            },
            index=pd.DatetimeIndex(timestamps),
        )


class KronosConsecutiveAdapterTest(unittest.TestCase):
    def test_predict_maps_consecutive_path_into_step_directions(self) -> None:
        candles = [
            build_candle(0, "98"),
            build_candle(1, "99"),
            build_candle(2, "100"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model"
            tokenizer_path = tmp_path / "tokenizer"
            model_path.mkdir()
            tokenizer_path.mkdir()

            adapter = KronosConsecutiveAdapter(
                model_path=model_path,
                predlen=2,
                lookback=3,
                logger=None,
                tokenizer_path=tokenizer_path,
                device="cpu",
                clip=5.0,
                direction_mode="close_to_close",
                stabilize_output=False,
                stability_context_path=None,
            )

            with patch("interval_bot.models.load_kronos_predictor", return_value=_FakePredictor()):
                prediction = adapter.predict(candles)

        self.assertEqual(len(prediction.steps), 2)
        self.assertEqual(prediction.steps[0].direction, SignalDirection.UP)
        self.assertEqual(prediction.steps[1].direction, SignalDirection.DOWN)
        self.assertAlmostEqual(prediction.steps[0].metadata["reference_close"], 100.0)
        self.assertAlmostEqual(prediction.steps[1].metadata["reference_close"], 101.0)


if __name__ == "__main__":
    unittest.main()
