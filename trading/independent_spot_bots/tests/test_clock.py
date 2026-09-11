from __future__ import annotations

import logging
import time
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interval_bot.logging_utils import ExchangeAlignedFormatter
from interval_bot.types import set_time_offset_ms, utc_now_ms


class ExchangeClockTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_time_offset_ms(0)

    def test_utc_now_ms_applies_exchange_offset(self) -> None:
        local_before = int(time.time() * 1000)
        set_time_offset_ms(2500)
        aligned_now = utc_now_ms()
        self.assertGreaterEqual(aligned_now - local_before, 2400)
        self.assertLessEqual(aligned_now - local_before, 2600)

    def test_logging_formatter_uses_exchange_offset(self) -> None:
        formatter = ExchangeAlignedFormatter("%(asctime)s", "%Y-%m-%dT%H:%M:%S%z")
        record = logging.makeLogRecord(
            {
                "name": "interval_bot.test",
                "levelno": logging.INFO,
                "levelname": "INFO",
                "msg": "hello",
                "created": 1_776_276_840.0,
            }
        )
        set_time_offset_ms(14_400_000)
        self.assertEqual(formatter.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "2026-04-15T22:14:00+0000")


if __name__ == "__main__":
    unittest.main()
