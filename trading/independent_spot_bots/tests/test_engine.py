from __future__ import annotations

import asyncio
import logging
import unittest
from decimal import Decimal
from types import SimpleNamespace

from interval_bot.engine import SpotIntervalBot
from interval_bot.exchange import BinanceSpotGateway
from interval_bot.types import ManagedOrder, OrderSide, Regime


class _FakeState:
    def __init__(self) -> None:
        self.regime = Regime.CASH
        self.clear_count = 0

    def clear_open_orders(self) -> None:
        self.clear_count += 1


class _FakeStateStore:
    def save(self, state) -> None:  # noqa: ANN001
        return None


class _ReplaceGateway:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def get_order(self, symbol: str, order_id: int) -> ManagedOrder:
        return ManagedOrder(
            order_id=order_id,
            client_order_id="cid",
            side=OrderSide.BUY,
            status="NEW",
            price=Decimal("74000"),
            orig_qty=Decimal("0.001"),
            executed_qty=Decimal("0"),
            raw={},
        )

    def cancel_order(self, symbol: str, order_id: int) -> None:
        self.cancel_calls += 1


class EngineTests(unittest.TestCase):
    def test_wait_for_fill_can_trigger_cancel_replace_before_full_timeout(self) -> None:
        bot = SpotIntervalBot.__new__(SpotIntervalBot)
        bot.config = SimpleNamespace(
            symbol="BTCFDUSD",
            cancel_replace_interval_seconds=0,
            order_status_poll_seconds=0,
        )
        bot.gateway = _ReplaceGateway()
        bot.state = _FakeState()
        bot.state_store = _FakeStateStore()
        bot.stop_event = asyncio.Event()
        bot.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

        order = ManagedOrder(
            order_id=123,
            client_order_id="cid",
            side=OrderSide.BUY,
            status="NEW",
            price=Decimal("74000"),
            orig_qty=Decimal("0.001"),
            executed_qty=Decimal("0"),
            raw={},
        )

        async def run_case() -> None:
            outcome = await SpotIntervalBot._wait_for_fill(
                bot,
                order=order,
                desired_regime=Regime.LONG,
                close_time=0,
                overall_deadline=asyncio.get_running_loop().time() + 5,
                replace_allowed=True,
            )
            self.assertEqual(outcome, "replace")

        asyncio.run(run_case())
        self.assertEqual(bot.gateway.cancel_calls, 1)
        self.assertEqual(bot.state.clear_count, 1)

    def test_wait_for_next_closed_candle_can_run_polling_without_websocket(self) -> None:
        gateway = BinanceSpotGateway(
            config=SimpleNamespace(
                api_key_env="MISSING_KEY",
                api_secret_env="MISSING_SECRET",
                use_official_sdk_public_endpoints=False,
                rest_base_url="https://api.binance.com",
                ws_base_url="wss://stream.binance.com:9443/ws",
                time_sync_interval_seconds=1800,
                use_websocket_close_detection=False,
            ),
            logger=logging.getLogger("interval_bot.test"),
        )

        def fail_if_called(symbol: str, interval: str) -> None:
            raise AssertionError("websocket stream should not be started in REST-only mode")

        gateway._ensure_stream = fail_if_called  # type: ignore[method-assign]
        gateway.get_latest_closed_candle = lambda symbol, interval: SimpleNamespace(close_time=123)  # type: ignore[method-assign]

        async def run_case() -> None:
            close_time = await gateway.wait_for_next_closed_candle(
                symbol="BTCFDUSD",
                interval="1d",
                after_close_time=100,
                polling_fallback_enabled=True,
                poll_interval_seconds=0,
                stop_event=asyncio.Event(),
            )
            self.assertEqual(close_time, 123)

        asyncio.run(run_case())


if __name__ == "__main__":
    unittest.main()
