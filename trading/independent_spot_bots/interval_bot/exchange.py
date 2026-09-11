from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import random
import time
import urllib.parse
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any

import requests
import websockets

from .config import BotConfig
from .types import (
    BalanceSnapshot,
    Candle,
    ManagedOrder,
    OrderSide,
    SignalDirection,
    SymbolFilters,
    set_time_offset_ms,
    decimal_from,
)


class BinanceAPIError(RuntimeError):
    pass


class BinanceSpotGateway:
    def __init__(self, config: BotConfig, logger) -> None:
        self.config = config
        self.logger = logger.getChild("exchange")
        self.api_key = os.getenv(config.api_key_env, "").strip()
        self.api_secret = os.getenv(config.api_secret_env, "").strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "independent-spot-bots/1.0"})
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        self._time_offset_ms = 0
        self._last_time_sync_monotonic = 0.0
        self._stream_queue: asyncio.Queue[int] = asyncio.Queue()
        self._stream_task: asyncio.Task | None = None
        self._stream_symbol: str | None = None
        self._stream_interval: str | None = None
        self._shutdown = asyncio.Event()
        self._sdk_client = self._try_create_sdk_client()

    def _try_create_sdk_client(self):
        if not self.config.use_official_sdk_public_endpoints:
            return None
        try:
            from binance_common.configuration import ConfigurationRestAPI
            from binance_sdk_spot.spot import Spot

            configuration = ConfigurationRestAPI(
                api_key=self.api_key or None,
                api_secret=self.api_secret or None,
                base_path=self.config.rest_base_url,
            )
            return Spot(config_rest_api=configuration)
        except Exception as exc:
            self.logger.warning("Official SDK unavailable, using raw REST fallback: %s", exc)
            return None

    def _normalize_sdk_payload(self, payload: Any) -> Any:
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, list):
            return [self._normalize_sdk_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._normalize_sdk_payload(item) for item in payload]
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "to_dict"):
            return payload.to_dict()
        if hasattr(payload, "__dict__"):
            return {
                key: self._normalize_sdk_payload(value)
                for key, value in payload.__dict__.items()
                if not key.startswith("_")
            }
        return payload

    def sync_time_offset(self) -> None:
        sample_offsets: list[int] = []
        for _ in range(3):
            local_before = int(time.time() * 1000)
            response = self._request("GET", "/api/v3/time", signed=False)
            local_after = int(time.time() * 1000)
            server_time = int(response["serverTime"])
            estimated_local = (local_before + local_after) // 2
            sample_offsets.append(server_time - estimated_local)
        self._time_offset_ms = int(sum(sample_offsets) / len(sample_offsets))
        self._last_time_sync_monotonic = time.monotonic()
        set_time_offset_ms(self._time_offset_ms)
        self.logger.info("Synchronized server time offset_ms=%s", self._time_offset_ms)

    def server_time_ms(self) -> int:
        if time.monotonic() - self._last_time_sync_monotonic > self.config.time_sync_interval_seconds:
            self.sync_time_offset()
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> Any:
        params = dict(params or {})
        url = f"{self.config.rest_base_url.rstrip('/')}{path}"
        attempts = 5

        for attempt in range(1, attempts + 1):
            try:
                request_params = dict(params)
                if signed:
                    if not self.api_key or not self.api_secret:
                        raise BinanceAPIError(
                            f"Missing credentials. Set {self.config.api_key_env} and {self.config.api_secret_env}."
                        )
                    request_params["timestamp"] = self.server_time_ms()
                    request_params["recvWindow"] = self.config.recv_window_ms
                    query_string = urllib.parse.urlencode(request_params, doseq=True)
                    signature = hmac.new(
                        self.api_secret.encode("utf-8"),
                        query_string.encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                    request_params["signature"] = signature

                response = self.session.request(
                    method,
                    url,
                    params=request_params,
                    timeout=self.config.request_timeout_seconds,
                )
                if response.status_code >= 500 or response.status_code in {418, 429}:
                    raise BinanceAPIError(f"Temporary Binance error {response.status_code}: {response.text}")
                if response.status_code >= 400:
                    error_payload = response.json() if response.text else {}
                    if error_payload.get("code") == -1021 and attempt < attempts:
                        self.logger.warning("Timestamp drift detected. Resyncing time and retrying.")
                        self.sync_time_offset()
                        continue
                    raise BinanceAPIError(
                        f"Binance request failed {response.status_code}: {response.text}"
                    )
                return response.json()
            except (requests.ConnectionError, requests.Timeout, BinanceAPIError) as exc:
                if attempt >= attempts:
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 15) + random.random()
                self.logger.warning(
                    "REST attempt %s/%s failed for %s %s: %s. Retrying in %.1fs.",
                    attempt,
                    attempts,
                    method,
                    path,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise BinanceAPIError(f"Failed request after retries: {method} {path}")

    def get_exchange_info(self, symbol: str) -> dict:
        if self._sdk_client is not None:
            try:
                response = self._sdk_client.rest_api.exchange_info(symbol=symbol)
                return self._normalize_sdk_payload(response.data())
            except Exception as exc:
                self.logger.warning("Official SDK exchange_info failed. Falling back to REST: %s", exc)
        return self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)

    def get_klines(self, symbol: str, interval: str, limit: int, end_time: int | None = None) -> list:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time

        if self._sdk_client is not None:
            try:
                sdk_params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
                if end_time is not None:
                    sdk_params["end_time"] = end_time
                response = self._sdk_client.rest_api.klines(**sdk_params)
                return self._normalize_sdk_payload(response.data())
            except Exception as exc:
                self.logger.warning("Official SDK klines failed. Falling back to REST: %s", exc)
        return self._request("GET", "/api/v3/klines", params, signed=False)

    def parse_candle(self, raw: list | dict) -> Candle:
        if isinstance(raw, dict):
            kline = raw.get("k", raw)
            return Candle(
                open_time=int(kline["t"]),
                open=decimal_from(kline["o"]),
                high=decimal_from(kline["h"]),
                low=decimal_from(kline["l"]),
                close=decimal_from(kline["c"]),
                volume=decimal_from(kline["v"]),
                close_time=int(kline["T"]),
                quote_asset_volume=decimal_from(kline["q"]),
                number_of_trades=int(kline["n"]),
                taker_buy_base_volume=decimal_from(kline["V"]),
                taker_buy_quote_volume=decimal_from(kline["Q"]),
                is_closed=bool(kline.get("x", True)),
            )

        return Candle(
            open_time=int(raw[0]),
            open=decimal_from(raw[1]),
            high=decimal_from(raw[2]),
            low=decimal_from(raw[3]),
            close=decimal_from(raw[4]),
            volume=decimal_from(raw[5]),
            close_time=int(raw[6]),
            quote_asset_volume=decimal_from(raw[7]),
            number_of_trades=int(raw[8]),
            taker_buy_base_volume=decimal_from(raw[9]),
            taker_buy_quote_volume=decimal_from(raw[10]),
            is_closed=True,
        )

    def get_closed_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
        end_close_time: int | None = None,
    ) -> list[Candle]:
        raw_candles = self.get_klines(symbol, interval, limit=min(limit + 2, 1000), end_time=end_close_time)
        candles = [self.parse_candle(item) for item in raw_candles]
        server_time = self.server_time_ms()
        closed = [
            candle
            for candle in candles
            if candle.close_time <= (end_close_time or server_time) and candle.close_time < server_time + 1
        ]
        return closed[-limit:]

    def get_closed_window(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        end_close_time: int | None = None,
    ) -> list[Candle]:
        candles = self.get_closed_candles(symbol, interval, limit=lookback + 8, end_close_time=end_close_time)
        if len(candles) < lookback:
            raise BinanceAPIError(
                f"Expected {lookback} closed candles for {symbol} {interval}, got {len(candles)}"
            )
        return candles[-lookback:]

    def get_latest_closed_candle(self, symbol: str, interval: str) -> Candle:
        candles = self.get_closed_candles(symbol, interval, limit=2)
        if not candles:
            raise BinanceAPIError(f"No closed candles returned for {symbol} {interval}")
        return candles[-1]

    @staticmethod
    def _field(payload: dict, *keys: str, default: Any = None) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _normalize_filter_entry(payload: dict) -> dict:
        actual = payload.get("actual_instance")
        if isinstance(actual, dict):
            return actual
        return payload

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        info = self.get_exchange_info(symbol)
        symbols = info["symbols"] if "symbols" in info else [info]
        symbol_info = next(item for item in symbols if self._field(item, "symbol") == symbol)
        normalized_filters = [
            self._normalize_filter_entry(item)
            for item in self._field(symbol_info, "filters", default=[])
        ]
        filter_map = {
            str(filter_type): item
            for item in normalized_filters
            if (filter_type := self._field(item, "filterType", "filter_type")) is not None
        }
        notional_filter = filter_map.get("NOTIONAL")
        min_notional_filter = filter_map.get("MIN_NOTIONAL", {})
        min_notional = self._field(min_notional_filter, "minNotional", "min_notional")
        if notional_filter and self._field(notional_filter, "minNotional", "min_notional"):
            min_notional = self._field(notional_filter, "minNotional", "min_notional")
        return SymbolFilters(
            symbol=symbol,
            base_asset=str(self._field(symbol_info, "baseAsset", "base_asset")),
            quote_asset=str(self._field(symbol_info, "quoteAsset", "quote_asset")),
            tick_size=decimal_from(self._field(filter_map["PRICE_FILTER"], "tickSize", "tick_size")),
            step_size=decimal_from(self._field(filter_map["LOT_SIZE"], "stepSize", "step_size")),
            min_qty=decimal_from(self._field(filter_map["LOT_SIZE"], "minQty", "min_qty")),
            max_qty=decimal_from(self._field(filter_map["LOT_SIZE"], "maxQty", "max_qty")),
            min_notional=decimal_from(min_notional or "0"),
            max_notional=(
                decimal_from(self._field(notional_filter, "maxNotional", "max_notional"))
                if notional_filter and self._field(notional_filter, "maxNotional", "max_notional")
                else None
            ),
            price_precision=self._field(symbol_info, "quotePrecision", "quote_precision"),
            quantity_precision=self._field(symbol_info, "baseAssetPrecision", "base_asset_precision"),
        )

    def get_book_ticker(self, symbol: str) -> tuple[Decimal, Decimal]:
        data = self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol}, signed=False)
        return decimal_from(data["bidPrice"]), decimal_from(data["askPrice"])

    def get_account_balances(self, base_asset: str, quote_asset: str) -> BalanceSnapshot:
        data = self._request("GET", "/api/v3/account", signed=True)
        balances = {item["asset"]: item for item in data["balances"]}
        base = balances.get(base_asset, {"free": "0", "locked": "0"})
        quote = balances.get(quote_asset, {"free": "0", "locked": "0"})
        return BalanceSnapshot(
            base_free=decimal_from(base["free"]),
            base_locked=decimal_from(base["locked"]),
            quote_free=decimal_from(quote["free"]),
            quote_locked=decimal_from(quote["locked"]),
            captured_at_ms=self.server_time_ms(),
        )

    def list_open_orders(self, symbol: str) -> list[ManagedOrder]:
        data = self._request("GET", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        return [self._parse_order(item) for item in data]

    def get_order(self, symbol: str, order_id: int) -> ManagedOrder:
        data = self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)
        return self._parse_order(data)

    def cancel_order(self, symbol: str, order_id: int) -> ManagedOrder:
        data = self._request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)
        return self._parse_order(data)

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> ManagedOrder:
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": self.decimal_to_str(price),
            "quantity": self.decimal_to_str(quantity),
            "newClientOrderId": client_order_id,
        }
        data = self._request("POST", "/api/v3/order", params, signed=True)
        return self._parse_order(data)

    def _parse_order(self, payload: dict) -> ManagedOrder:
        return ManagedOrder(
            order_id=int(payload["orderId"]),
            client_order_id=str(payload.get("clientOrderId", "")),
            side=OrderSide(payload["side"]),
            status=str(payload["status"]),
            price=decimal_from(payload.get("price", "0")),
            orig_qty=decimal_from(payload.get("origQty", "0")),
            executed_qty=decimal_from(payload.get("executedQty", "0")),
            raw=dict(payload),
        )

    async def wait_for_next_closed_candle(
        self,
        symbol: str,
        interval: str,
        after_close_time: int,
        polling_fallback_enabled: bool,
        poll_interval_seconds: int,
        stop_event: asyncio.Event,
    ) -> int | None:
        if self.config.use_websocket_close_detection:
            self._ensure_stream(symbol, interval)
        else:
            self.logger.info("Using REST polling for candle-close detection on %s %s", symbol, interval)
        timeout_seconds = poll_interval_seconds if polling_fallback_enabled else 60

        while not stop_event.is_set():
            try:
                close_time = await asyncio.wait_for(self._stream_queue.get(), timeout=timeout_seconds)
                if close_time > after_close_time:
                    return close_time
            except asyncio.TimeoutError:
                if not polling_fallback_enabled:
                    continue
                latest = self.get_latest_closed_candle(symbol, interval)
                if latest.close_time > after_close_time:
                    self.logger.warning("Falling back to REST candle-close detection for %s %s", symbol, interval)
                    return latest.close_time
        return None

    def _ensure_stream(self, symbol: str, interval: str) -> None:
        if (
            self._stream_task is not None
            and not self._stream_task.done()
            and self._stream_symbol == symbol
            and self._stream_interval == interval
        ):
            return

        self._stream_symbol = symbol
        self._stream_interval = interval
        self._stream_task = asyncio.create_task(self._run_stream(symbol, interval))

    async def _run_stream(self, symbol: str, interval: str) -> None:
        stream_name = f"{symbol.lower()}@kline_{interval}"
        ws_url = f"{self.config.ws_base_url.rstrip('/')}/{stream_name}"
        backoff = 1.0

        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=60,
                    open_timeout=15,
                    close_timeout=10,
                ) as websocket:
                    self.logger.info("Connected websocket stream=%s", stream_name)
                    backoff = 1.0
                    async for message in websocket:
                        payload = json.loads(message)
                        if "data" in payload:
                            payload = payload["data"]
                        if payload.get("e") != "kline":
                            continue
                        kline = payload.get("k", {})
                        if kline.get("x") is True:
                            close_time = int(kline["T"])
                            await self._stream_queue.put(close_time)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._shutdown.is_set():
                    break
                self.logger.warning(
                    "Websocket disconnected for %s %s: %s. Reconnecting in %.1fs.",
                    symbol,
                    interval,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def quantize_price(self, value: Decimal, tick_size: Decimal, side: OrderSide) -> Decimal:
        rounding = ROUND_DOWN if side == OrderSide.BUY else ROUND_UP
        return (value / tick_size).to_integral_value(rounding=rounding) * tick_size

    def quantize_qty(self, value: Decimal, step_size: Decimal) -> Decimal:
        return (value / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size

    @staticmethod
    def decimal_to_str(value: Decimal) -> str:
        normalized = value.normalize()
        return format(normalized, "f")

    async def close(self) -> None:
        self._shutdown.set()
        if self._stream_task is not None:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        self.session.close()
