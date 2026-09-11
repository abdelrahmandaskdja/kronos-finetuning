from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interval_bot.state import BotState, StateStore
from interval_bot.types import BalanceSnapshot, PredictionStep, Regime, SignalDirection


class StatePersistenceTest(unittest.TestCase):
    def test_round_trip_state_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            store = StateStore(path)

            state = BotState(
                last_processed_candle_close_time=1713139200000,
                regime=Regime.LONG,
                open_order_ids=[101, 202],
                last_balances=BalanceSnapshot(
                    base_free=Decimal("0.1234"),
                    base_locked=Decimal("0.0100"),
                    quote_free=Decimal("456.78"),
                    quote_locked=Decimal("12.34"),
                    captured_at_ms=1713139201000,
                ),
                forecast_steps=[
                    PredictionStep(direction=SignalDirection.UP, score=0.5, raw_value=0.5),
                    PredictionStep(direction=SignalDirection.DOWN, score=-0.25, raw_value=-0.25),
                ],
                forecast_index=1,
                forecast_cycle_started_at=1713135600000,
                updated_at_ms=1713139202000,
            )

            store.save(state)
            loaded = store.load()

            self.assertEqual(loaded.last_processed_candle_close_time, state.last_processed_candle_close_time)
            self.assertEqual(loaded.regime, Regime.LONG)
            self.assertEqual(loaded.open_order_ids, [101, 202])
            self.assertEqual(loaded.last_balances.base_free, Decimal("0.1234"))
            self.assertEqual(loaded.last_balances.quote_free, Decimal("456.78"))
            self.assertEqual(len(loaded.forecast_steps), 2)
            self.assertEqual(loaded.forecast_steps[0].direction, SignalDirection.UP)
            self.assertEqual(loaded.forecast_index, 1)
            self.assertEqual(loaded.forecast_cycle_started_at, 1713135600000)


if __name__ == "__main__":
    unittest.main()
