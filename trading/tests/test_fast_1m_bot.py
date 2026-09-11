import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_PATH = REPO_ROOT / "trading" / "binance_spot_multi_interval_bot.py"


def load_bot_module():
    spec = importlib.util.spec_from_file_location("trading_fast_1m_bot", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Fast1mBotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot_module()

    def test_ws_signature_payload_uses_sorted_keys(self):
        payload = self.bot.build_ws_api_signature_payload(
            {
                "timestamp": 123,
                "symbol": "BTCFDUSD",
                "side": "BUY",
                "apiKey": "abc",
            }
        )
        self.assertEqual(payload, "apiKey=abc&side=BUY&symbol=BTCFDUSD&timestamp=123")

    def test_resolve_runtime_device_preserves_explicit_cpu(self):
        self.assertEqual(self.bot.resolve_runtime_device("cpu"), "cpu")

    def test_exchange_utc_now_ms_uses_synced_offset(self):
        self.bot.update_exchange_time_offset(2500, synced_local_ms=self.bot.utc_now_ms())
        local_before = self.bot.utc_now_ms()
        exchange_now_ms = self.bot.exchange_utc_now_ms()
        self.bot.update_exchange_time_offset(0, synced_local_ms=self.bot.utc_now_ms())
        self.assertGreaterEqual(exchange_now_ms - local_before, 2400)
        self.assertLessEqual(exchange_now_ms - local_before, 2600)

    def test_compute_fast_1m_forecast_step_index_advances_per_bar(self):
        step0 = self.bot.compute_fast_1m_forecast_step_index(
            cycle_context_timestamp="2026-04-15T15:17:00",
            current_context_timestamp=self.bot.pd.Timestamp("2026-04-15T15:17:00"),
            interval="1m",
        )
        step5 = self.bot.compute_fast_1m_forecast_step_index(
            cycle_context_timestamp="2026-04-15T15:17:00",
            current_context_timestamp=self.bot.pd.Timestamp("2026-04-15T15:22:00"),
            interval="1m",
        )
        self.assertEqual(step0, 0)
        self.assertEqual(step5, 5)

    def test_build_fast_1m_signal_from_cycle_uses_selected_row(self):
        spec = {
            "model_id": "1m_test",
            "model_name": "1m test model",
            "family": "consecutive_rollout",
            "role": "micro",
            "metric_name": "pred_close_vs_prev_pred_close",
            "interval": "1m",
            "binance_interval": "1m",
        }
        cycle = {
            "context_timestamp": "2026-04-15T15:17:00",
            "rows": [
                {
                    "timestamp": "2026-04-15T15:18:00",
                    "pred_dir": 1,
                    "move_pct": 0.25,
                    "reference_close": 74200.0,
                    "close": 74210.0,
                },
                {
                    "timestamp": "2026-04-15T15:19:00",
                    "pred_dir": 0,
                    "move_pct": -0.15,
                    "reference_close": 74210.0,
                    "close": 74198.0,
                },
            ],
        }
        signal = self.bot.build_fast_1m_signal_from_cycle(
            spec=spec,
            cycle=cycle,
            step_index=1,
            current_context_timestamp=self.bot.pd.Timestamp("2026-04-15T15:18:00"),
            market_path=Path("/tmp/fake_market.csv"),
        )
        self.assertEqual(signal.direction, "DOWN")
        self.assertEqual(signal.context_timestamp, "2026-04-15T15:18:00")
        self.assertEqual(signal.target_timestamp, "2026-04-15T15:19:00")
        self.assertEqual(signal.reference_price, 74210.0)

    def test_upsert_fast_closed_bar_preserves_timestamp_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "market.csv"
            initial = self.bot.pd.DataFrame(
                [
                    {
                        "timestamp": self.bot.pd.Timestamp("2026-04-15T15:17:00"),
                        "open": 74000.0,
                        "close": 74010.0,
                        "high": 74020.0,
                        "low": 73990.0,
                        "volume": 1.0,
                        "amount": 74010.0,
                    }
                ]
            )
            self.bot.write_market_dataframe(path, initial)
            self.bot.upsert_fast_closed_bar(
                path=path,
                closed_bar={
                    "open_time_ms": int(self.bot.pd.Timestamp("2026-04-15T15:18:00").timestamp() * 1000),
                    "open": 74010.0,
                    "close": 74030.0,
                    "high": 74035.0,
                    "low": 74005.0,
                    "volume": 0.5,
                    "amount": 37015.0,
                },
                keep_rows=16,
            )
            reloaded = self.bot.load_market_dataframe(path)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(str(reloaded["timestamp"].iloc[-1]), "2026-04-15 15:18:00")

    def test_primary_peg_limit_maker_plan_omits_explicit_price(self):
        config = self.bot.build_default_config()
        config["symbol"] = "BTCFDUSD"
        config["execution"]["use_primary_peg_limit_maker"] = True
        config["risk"]["max_trade_notional_usdt"] = 30.0
        signal_summary = {"target_allocation": 1.0, "target_exposure": 1.0}
        symbol_rules = {
            "symbol": "BTCFDUSD",
            "base_asset": "BTC",
            "quote_asset": "FDUSD",
            "filters": {
                "LOT_SIZE": {"stepSize": "0.00001", "minQty": "0.00001"},
                "PRICE_FILTER": {"tickSize": "0.01"},
                "NOTIONAL": {"minNotional": "5"},
            },
        }
        balances = {
            "BTC": {"free": self.bot.Decimal("0"), "total": self.bot.Decimal("0")},
            "FDUSD": {"free": self.bot.Decimal("30"), "total": self.bot.Decimal("30")},
        }
        book_ticker = {
            "bidPrice": "74000.00",
            "bidQty": "1.0",
            "askPrice": "74000.01",
            "askQty": "1.0",
        }

        plan = self.bot.plan_spot_maker_limit_rebalance(
            config=config,
            signal_summary=signal_summary,
            symbol_rules=symbol_rules,
            balances=balances,
            current_price=self.bot.Decimal("74000.00"),
            book_ticker=book_ticker,
        )

        self.assertEqual(plan["order_style"], "maker_limit")
        self.assertTrue(plan["use_primary_peg_limit_maker"])
        self.assertIn("order_params", plan)
        self.assertEqual(plan["order_params"]["type"], "LIMIT_MAKER")
        self.assertEqual(plan["order_params"]["pegPriceType"], "PRIMARY_PEG")
        self.assertNotIn("price", plan["order_params"])
        self.assertGreater(plan["limit_quantity"], 0.0)

    def test_build_live_spot_client_uses_ws_order_entry_when_enabled(self):
        config = self.bot.build_default_config()
        fake_rest_client = object()
        fake_ws_client = object()

        with (
            patch.dict(self.bot.os.environ, {"BINANCE_API_KEY": "key", "BINANCE_API_SECRET": "secret"}, clear=False),
            patch.object(self.bot, "BinanceSpotClient", return_value=fake_rest_client) as rest_ctor,
            patch.object(self.bot, "BinanceSpotWsApiClient", return_value=fake_ws_client) as ws_ctor,
        ):
            client = self.bot.build_live_spot_client(config=config, mode="live", enable_ws_orders=True)

        self.assertIsInstance(client, self.bot.HybridBinanceSpotClient)
        self.assertIs(client.rest_client, fake_rest_client)
        self.assertIs(client.ws_order_client, fake_ws_client)
        rest_ctor.assert_called_once()
        ws_ctor.assert_called_once()

    def test_resolve_interval_forecast_cycle_signal_reuses_saved_cycle(self):
        spec = {
            "model_id": "1h_test",
            "model_name": "1h test model",
            "family": "consecutive_rollout",
            "role": "confirm",
            "metric_name": "pred_close_vs_prev_pred_close",
            "interval": "1h",
            "binance_interval": "1h",
        }
        cycle = {
            "metric_name": "pred_close_vs_prev_pred_close",
            "context_timestamp": "2026-04-15T12:00:00",
            "binance_interval": "1h",
            "rows": [
                {
                    "timestamp": "2026-04-15T13:00:00",
                    "pred_dir": 1,
                    "move_pct": 0.5,
                    "reference_close": 74200.0,
                    "close": 74250.0,
                },
                {
                    "timestamp": "2026-04-15T14:00:00",
                    "pred_dir": 0,
                    "move_pct": -0.2,
                    "reference_close": 74250.0,
                    "close": 74235.0,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            market_path = Path(tmpdir) / "market.csv"
            frame = self.bot.pd.DataFrame(
                [
                    {
                        "timestamp": self.bot.pd.Timestamp("2026-04-15T10:00:00"),
                        "open": 74000.0,
                        "close": 74010.0,
                        "high": 74020.0,
                        "low": 73990.0,
                        "volume": 1.0,
                        "amount": 74010.0,
                    },
                    {
                        "timestamp": self.bot.pd.Timestamp("2026-04-15T11:00:00"),
                        "open": 74010.0,
                        "close": 74100.0,
                        "high": 74110.0,
                        "low": 74000.0,
                        "volume": 1.0,
                        "amount": 74100.0,
                    },
                    {
                        "timestamp": self.bot.pd.Timestamp("2026-04-15T12:00:00"),
                        "open": 74100.0,
                        "close": 74200.0,
                        "high": 74210.0,
                        "low": 74090.0,
                        "volume": 1.0,
                        "amount": 74200.0,
                    },
                ]
            )
            self.bot.write_market_dataframe(market_path, frame)
            state = {}

            with patch.object(self.bot, "build_consecutive_forecast_cycle", return_value=cycle) as build_cycle:
                signal, diagnostics = self.bot.resolve_interval_forecast_cycle_signal(
                    state=state,
                    state_key="forecast_cycle_1h",
                    spec=spec,
                    market_path=market_path,
                )
                self.assertEqual(build_cycle.call_count, 1)
                self.assertTrue(diagnostics["refreshed"])
                self.assertEqual(diagnostics["step_index"], 0)
                self.assertEqual(signal.direction, "UP")

            frame = self.bot.pd.concat(
                [
                    frame,
                    self.bot.pd.DataFrame(
                        [
                            {
                                "timestamp": self.bot.pd.Timestamp("2026-04-15T13:00:00"),
                                "open": 74200.0,
                                "close": 74210.0,
                                "high": 74220.0,
                                "low": 74190.0,
                                "volume": 1.0,
                                "amount": 74210.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            self.bot.write_market_dataframe(market_path, frame)
            with patch.object(self.bot, "build_consecutive_forecast_cycle", side_effect=AssertionError("unexpected rebuild")):
                signal, diagnostics = self.bot.resolve_interval_forecast_cycle_signal(
                    state=state,
                    state_key="forecast_cycle_1h",
                    spec=spec,
                    market_path=market_path,
                )
                self.assertFalse(diagnostics["refreshed"])
                self.assertEqual(diagnostics["step_index"], 1)
                self.assertEqual(signal.direction, "DOWN")

    def test_should_run_same_bar_maintenance_only_when_live_maker_work_exists(self):
        config = self.bot.build_default_config()
        args = self.bot.argparse.Namespace(send_orders=True, test_order=False, poll_seconds=5)
        result = {
            "execution": {
                "kind": "spot",
                "managed_open_order_count": 1,
                "order_style": "maker_limit",
                "order_sent": False,
                "order_params": None,
                "side": "BUY",
                "order_reason": "rebalance_buy",
            }
        }
        self.assertTrue(self.bot.should_run_same_bar_maintenance(result=result, config=config, args=args))
        args = self.bot.argparse.Namespace(send_orders=False, test_order=False, poll_seconds=5)
        self.assertFalse(self.bot.should_run_same_bar_maintenance(result=result, config=config, args=args))

    def test_should_run_same_bar_maintenance_can_require_existing_order(self):
        config = self.bot.build_default_config()
        config["execution"]["same_bar_maintenance_requires_existing_order"] = True
        args = self.bot.argparse.Namespace(send_orders=True, test_order=False, poll_seconds=5)
        result = {
            "execution": {
                "kind": "spot",
                "managed_open_order_count": 0,
                "order_style": "maker_limit",
                "order_sent": False,
                "order_params": {
                    "symbol": "BTCFDUSD",
                    "side": "SELL",
                    "type": "LIMIT_MAKER",
                    "quantity": "0.00039",
                },
                "side": "SELL",
                "order_reason": None,
            }
        }
        self.assertFalse(self.bot.should_run_same_bar_maintenance(result=result, config=config, args=args))

        result["execution"]["managed_open_order_count"] = 1
        self.assertTrue(self.bot.should_run_same_bar_maintenance(result=result, config=config, args=args))

    def test_execute_maker_limit_plan_keeps_existing_same_side_order_when_balance_is_temporarily_locked(self):
        config = self.bot.build_default_config()
        config["symbol"] = "BTCFDUSD"
        config["execution"]["maker_client_order_prefix"] = "k1fdd"

        class FakeClient:
            def __init__(self):
                self.cancel_calls = []

            def get_open_orders(self, params):
                return [
                    {
                        "orderId": 123,
                        "clientOrderId": "k1fdd-B-keepme",
                        "side": "BUY",
                        "type": "LIMIT_MAKER",
                        "price": "75000.00",
                        "origQty": "0.00038",
                        "executedQty": "0.00000",
                        "time": 1,
                        "updateTime": 1,
                    }
                ]

            def cancel_order(self, params):
                self.cancel_calls.append(params)
                return {"status": "CANCELED", "orderId": params["orderId"]}

        client = FakeClient()
        plan = {
            "side": "BUY",
            "order_params": None,
            "order_reason": "buy_qty_below_lot_size",
            "tick_size": 0.01,
            "step_size": 0.00001,
        }

        result = self.bot.execute_maker_limit_plan(
            client=client,
            config=config,
            plan=plan,
            test_order=False,
        )

        self.assertEqual(result["action"], "keep_existing_limit_maker")
        self.assertEqual(len(result["cancel_responses"]), 0)
        self.assertEqual(len(client.cancel_calls), 0)


if __name__ == "__main__":
    unittest.main()
