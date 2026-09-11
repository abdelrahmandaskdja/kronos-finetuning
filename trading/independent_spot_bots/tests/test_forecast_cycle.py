from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interval_bot.state import BotState
from interval_bot.types import ModelPrediction, PredictionStep, SignalDirection


def build_prediction() -> ModelPrediction:
    return ModelPrediction(
        steps=[
            PredictionStep(direction=SignalDirection.UP if i % 2 == 0 else SignalDirection.DOWN, score=float(i))
            for i in range(16)
        ],
        generated_at_ms=123,
        source="test",
    )


class ForecastCycleTest(unittest.TestCase):
    def test_predlen_16_consumes_one_step_per_candle(self) -> None:
        state = BotState()
        prediction = build_prediction()
        state.replace_forecast(prediction, cycle_started_at=1000)

        consumed = []
        for _ in range(16):
            self.assertTrue(state.has_active_forecast())
            consumed.append(state.consume_next_forecast_step())

        self.assertEqual(len(consumed), 16)
        self.assertEqual(consumed[0].score, 0.0)
        self.assertEqual(consumed[-1].score, 15.0)
        self.assertFalse(state.has_active_forecast())
        self.assertEqual(state.forecast_index, 16)

    def test_new_cycle_resets_index_after_refresh(self) -> None:
        state = BotState()
        first = build_prediction()
        second = build_prediction()

        state.replace_forecast(first, cycle_started_at=1000)
        for _ in range(16):
            state.consume_next_forecast_step()

        self.assertFalse(state.has_active_forecast())
        state.replace_forecast(second, cycle_started_at=2000)

        self.assertTrue(state.has_active_forecast())
        self.assertEqual(state.forecast_index, 0)
        self.assertEqual(state.forecast_cycle_started_at, 2000)


if __name__ == "__main__":
    unittest.main()
