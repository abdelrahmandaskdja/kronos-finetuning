from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
from collections import deque
from datetime import UTC, datetime, timedelta
from time import monotonic
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config import Settings
from ..schemas import CandlePoint, LiveStatus, MarketSnapshot
from ..websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)
HTTP_HEADERS = {"User-Agent": "Marketiser/0.1"}
HISTORY_CACHE_MAX_ITEMS = 96
HISTORY_CACHE_TTL_RECENT_SECONDS = 8.0
HISTORY_CACHE_TTL_STABLE_SECONDS = 300.0


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def interval_to_timedelta(interval: str) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    return mapping.get(interval, timedelta(minutes=5))


class MarketDataService:
    def __init__(self, settings: Settings, ws_manager: WebSocketManager) -> None:
        self.settings = settings
        self.ws_manager = ws_manager
        self.snapshots: dict[str, MarketSnapshot] = {}
        self.candles: dict[tuple[str, str], deque[CandlePoint]] = {}
        self.history_cache: dict[tuple[str, str, int, int], tuple[float, list[CandlePoint]]] = {}
        self.live_status = LiveStatus(state="offline", detail="Booting market stream", updated_at=utc_now_iso())
        self._stream_task: asyncio.Task | None = None
        self._mock_task: asyncio.Task | None = None
        self._stopped = False

        for symbol in self.settings.symbols:
            for interval in self.settings.chart_intervals:
                self.candles[(symbol, interval)] = deque(maxlen=self.settings.max_candle_limit)

    async def start(self) -> None:
        await self.bootstrap()
        self._stream_task = asyncio.create_task(self._stream_loop(), name="marketiser-binance-rest-poller")

    async def stop(self) -> None:
        self._stopped = True
        if self._stream_task:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        if self._mock_task:
            self._mock_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._mock_task

    async def bootstrap(self) -> None:
        try:
            await self._refresh_snapshots(broadcast=False)
            await self._bootstrap_candles()
            await self._set_status("live", "Polling Binance REST market data")
        except Exception as exc:
            logger.warning("Binance REST bootstrap failed: %s", exc)
            self._seed_mock_state()
            await self._set_status("mock", f"Running with simulated market data after Binance REST bootstrap failure: {exc}")

    async def _stream_loop(self) -> None:
        failure_count = 0
        next_candle_refresh_at = monotonic()

        while not self._stopped:
            if not self.settings.enable_binance_rest:
                await self._start_mock_loop("Binance REST polling disabled")
                return
            try:
                include_candles = monotonic() >= next_candle_refresh_at
                await self._refresh_market_state(include_candles=include_candles, broadcast=True)
                if include_candles:
                    next_candle_refresh_at = monotonic() + self.settings.binance_candle_poll_seconds
                await self._stop_mock_loop()
                failure_count = 0
                detail = "Polling Binance REST market data"
                if include_candles:
                    detail = "Polling Binance REST market data and refreshing candles"
                await self._set_status("live", detail)
                await asyncio.sleep(max(self.settings.binance_snapshot_poll_seconds, 1.0))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure_count += 1
                logger.warning("Binance REST poll failed on attempt %s: %s", failure_count, exc)
                if failure_count >= 3:
                    await self._start_mock_loop(f"Live simulation active after Binance REST failures: {exc}")
                    await asyncio.sleep(max(self.settings.binance_snapshot_poll_seconds, 10.0))
                    continue
                await self._set_status("reconnecting", f"Binance REST retry {failure_count} in progress")
                await asyncio.sleep(min(self.settings.binance_snapshot_poll_seconds * failure_count, 15.0))

    async def _refresh_market_state(self, include_candles: bool, broadcast: bool) -> None:
        await self._refresh_snapshots(broadcast=broadcast)
        if include_candles:
            await self._refresh_recent_candles(broadcast=broadcast)

    async def _bootstrap_candles(self) -> None:
        async def fetch(symbol: str, interval: str) -> tuple[str, str, list[CandlePoint]]:
            rows = await asyncio.to_thread(
                self._fetch_recent_klines,
                symbol,
                interval,
                self.settings.bootstrap_candle_limit,
            )
            candles = [self._row_to_candle(symbol, interval, row) for row in rows]
            return symbol, interval, candles

        results = await asyncio.gather(
            *(fetch(symbol, interval) for symbol in self.settings.symbols for interval in self.settings.chart_intervals)
        )
        for symbol, interval, candles in results:
            bucket = self.candles[(symbol, interval)]
            bucket.clear()
            for candle in candles:
                bucket.append(candle)

    async def _refresh_snapshots(self, broadcast: bool) -> None:
        async def fetch(symbol: str) -> MarketSnapshot:
            payload = await asyncio.to_thread(self._fetch_symbol_ticker, symbol)
            return MarketSnapshot(
                symbol=symbol,
                last_price=float(payload["lastPrice"]),
                price_change_pct=float(payload["priceChangePercent"]),
                volume=float(payload["volume"]),
                quote_volume=float(payload["quoteVolume"]),
                high_24h=float(payload["highPrice"]),
                low_24h=float(payload["lowPrice"]),
                open_24h=float(payload["openPrice"]),
                updated_at=utc_now_iso(),
            )

        snapshots = await asyncio.gather(*(fetch(symbol) for symbol in self.settings.symbols))
        for snapshot in snapshots:
            previous = self.snapshots.get(snapshot.symbol)
            self.snapshots[snapshot.symbol] = snapshot
            if broadcast and self._snapshot_changed(previous, snapshot):
                await self.ws_manager.broadcast({"type": "market.snapshot", "payload": snapshot.model_dump()})

    async def _refresh_recent_candles(self, broadcast: bool) -> None:
        async def fetch(symbol: str, interval: str) -> tuple[str, str, list[CandlePoint]]:
            rows = await asyncio.to_thread(self._fetch_recent_klines, symbol, interval, 3)
            candles = [self._row_to_candle(symbol, interval, row) for row in rows]
            return symbol, interval, candles

        results = await asyncio.gather(
            *(fetch(symbol, interval) for symbol in self.settings.symbols for interval in self.settings.chart_intervals)
        )

        for symbol, interval, candles in results:
            for candle in candles:
                self._upsert_candle(candle)
            if broadcast and candles:
                await self.ws_manager.broadcast({"type": "market.kline", "payload": candles[-1].model_dump()})

    def _fetch_symbol_ticker(self, symbol: str) -> dict:
        url = f"{self.settings.binance_rest_url}/api/v3/ticker/24hr?{urlencode({'symbol': symbol})}"
        payload = self._fetch_json(url)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected ticker response for {symbol}")
        return payload

    def _fetch_recent_klines(self, symbol: str, interval: str, limit: int) -> list[list]:
        url = f"{self.settings.binance_rest_url}/api/v3/klines?{urlencode({'symbol': symbol, 'interval': interval, 'limit': limit})}"
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected kline response for {symbol} {interval}")
        return payload

    def _fetch_klines_window(
        self,
        symbol: str,
        interval: str,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        params: dict[str, int | str] = {
            "symbol": symbol,
            "interval": interval,
            "limit": max(1, min(limit, 1000)),
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        url = f"{self.settings.binance_rest_url}/api/v3/klines?{urlencode(params)}"
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected kline response for {symbol} {interval}")
        return payload

    def _history_cache_ttl(self, *, interval: str, to_seconds: int) -> float:
        interval_seconds = max(int(interval_to_timedelta(interval).total_seconds()), 60)
        now_seconds = int(datetime.now(tz=UTC).timestamp())
        if to_seconds >= now_seconds - (interval_seconds * 2):
            return HISTORY_CACHE_TTL_RECENT_SECONDS
        return HISTORY_CACHE_TTL_STABLE_SECONDS

    def _get_cached_history(
        self,
        *,
        symbol: str,
        interval: str,
        from_seconds: int,
        to_seconds: int,
    ) -> list[CandlePoint] | None:
        key = (symbol, interval, from_seconds, to_seconds)
        cached = self.history_cache.get(key)
        if not cached:
            return None
        cached_at, candles = cached
        ttl = self._history_cache_ttl(interval=interval, to_seconds=to_seconds)
        if monotonic() - cached_at > ttl:
            self.history_cache.pop(key, None)
            return None
        return candles

    def _store_cached_history(
        self,
        *,
        symbol: str,
        interval: str,
        from_seconds: int,
        to_seconds: int,
        candles: list[CandlePoint],
    ) -> list[CandlePoint]:
        key = (symbol, interval, from_seconds, to_seconds)
        sorted_candles = self._sorted_candles(candles)
        self.history_cache[key] = (monotonic(), sorted_candles)
        if len(self.history_cache) > HISTORY_CACHE_MAX_ITEMS:
            stale_keys = sorted(self.history_cache.items(), key=lambda item: item[1][0])[: len(self.history_cache) - HISTORY_CACHE_MAX_ITEMS]
            for stale_key, _ in stale_keys:
                self.history_cache.pop(stale_key, None)
        return sorted_candles

    def _row_to_candle(self, symbol: str, interval: str, row: list) -> CandlePoint:
        return CandlePoint(
            symbol=symbol,
            interval=interval,
            open_time=ms_to_iso(int(row[0])),
            close_time=ms_to_iso(int(row[6])),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            is_closed=datetime.now(tz=UTC) >= datetime.fromtimestamp(int(row[6]) / 1000, tz=UTC),
        )

    def _candle_open_ms(self, candle: CandlePoint) -> int:
        return int(datetime.fromisoformat(candle.open_time).timestamp() * 1000)

    def _sorted_candles(self, candles: list[CandlePoint]) -> list[CandlePoint]:
        deduped: dict[int, CandlePoint] = {}
        for candle in candles:
            try:
                deduped[self._candle_open_ms(candle)] = candle
            except (TypeError, ValueError, OverflowError):
                continue
        return [candle for _, candle in sorted(deduped.items())]

    def _snapshot_changed(self, previous: MarketSnapshot | None, current: MarketSnapshot) -> bool:
        if previous is None:
            return True
        return previous.model_dump() != current.model_dump()

    def _fetch_json(self, url: str) -> dict | list:
        request = Request(url, headers=HTTP_HEADERS)
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    async def _set_status(self, state: str, detail: str) -> None:
        if self.live_status.state == state and self.live_status.detail == detail:
            return
        self.live_status = LiveStatus(state=state, detail=detail, updated_at=utc_now_iso())
        await self.ws_manager.broadcast({"type": "system.status", "payload": self.live_status.model_dump()})

    async def _stop_mock_loop(self) -> None:
        if not self._mock_task:
            return
        task = self._mock_task
        self._mock_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _seed_mock_state(self) -> None:
        base_prices = {
            "BTCUSDT": 93500.0,
        }
        for symbol in self.settings.symbols:
            base_price = base_prices.get(symbol, 100.0)
            self.snapshots[symbol] = MarketSnapshot(
                symbol=symbol,
                last_price=base_price,
                price_change_pct=0.0,
                volume=1000000.0,
                quote_volume=base_price * 1000,
                high_24h=base_price * 1.02,
                low_24h=base_price * 0.98,
                open_24h=base_price,
                updated_at=utc_now_iso(),
            )
            for interval in self.settings.chart_intervals:
                bucket = self.candles[(symbol, interval)]
                bucket.clear()
                delta = interval_to_timedelta(interval)
                now = datetime.now(tz=UTC)
                price = base_price
                for index in range(self.settings.bootstrap_candle_limit):
                    timestamp = now - delta * (self.settings.bootstrap_candle_limit - index)
                    drift = math.sin(index / 6) * 0.001 + math.cos(index / 13) * 0.0006
                    next_price = price * (1 + drift)
                    high = max(price, next_price) * 1.0018
                    low = min(price, next_price) * 0.9982
                    bucket.append(
                        CandlePoint(
                            symbol=symbol,
                            interval=interval,
                            open_time=timestamp.isoformat(),
                            close_time=(timestamp + delta).isoformat(),
                            open=round(price, 6),
                            high=round(high, 6),
                            low=round(low, 6),
                            close=round(next_price, 6),
                            volume=round(500 + index * 2.5, 2),
                            is_closed=True,
                        )
                    )
                    price = next_price

    async def _start_mock_loop(self, detail: str) -> None:
        if not self.snapshots:
            self._seed_mock_state()
        if not self._mock_task:
            self._mock_task = asyncio.create_task(self._run_mock_loop(), name="marketiser-mock-stream")
        await self._set_status("mock", detail)

    async def _run_mock_loop(self) -> None:
        while not self._stopped:
            for symbol in self.settings.symbols:
                snapshot = self.snapshots[symbol]
                volatility = 0.0009 if symbol == "BTCUSDT" else 0.0014
                drift = random.gauss(0.0, volatility)
                next_price = max(snapshot.last_price * (1 + drift), 0.0001)
                open_24h = snapshot.open_24h or snapshot.last_price
                change_pct = ((next_price / open_24h) - 1) * 100
                updated_snapshot = snapshot.model_copy(
                    update={
                        "last_price": next_price,
                        "price_change_pct": change_pct,
                        "high_24h": max(snapshot.high_24h, next_price),
                        "low_24h": min(snapshot.low_24h, next_price),
                        "volume": snapshot.volume + random.uniform(1000, 7500),
                        "quote_volume": snapshot.quote_volume + next_price * random.uniform(100, 400),
                        "updated_at": utc_now_iso(),
                    }
                )
                self.snapshots[symbol] = updated_snapshot
                await self.ws_manager.broadcast({"type": "market.snapshot", "payload": updated_snapshot.model_dump()})

                for interval in self.settings.chart_intervals:
                    self._append_mock_candle(symbol, interval, next_price)
                    latest = self.candles[(symbol, interval)][-1]
                    await self.ws_manager.broadcast({"type": "market.kline", "payload": latest.model_dump()})
            await asyncio.sleep(1.5)

    def _append_mock_candle(self, symbol: str, interval: str, next_price: float) -> None:
        bucket = self.candles[(symbol, interval)]
        delta = interval_to_timedelta(interval)
        now = datetime.now(tz=UTC)
        previous_close = bucket[-1].close if bucket else next_price
        open_price = previous_close
        high = max(open_price, next_price) * 1.0015
        low = min(open_price, next_price) * 0.9985
        bucket.append(
            CandlePoint(
                symbol=symbol,
                interval=interval,
                open_time=(now - delta).isoformat(),
                close_time=now.isoformat(),
                open=round(open_price, 6),
                high=round(high, 6),
                low=round(low, 6),
                close=round(next_price, 6),
                volume=round(random.uniform(300, 1200), 2),
                is_closed=False,
            )
        )

    def _upsert_candle(self, candle: CandlePoint) -> None:
        bucket = self.candles[(candle.symbol, candle.interval)]
        for index in range(len(bucket) - 1, -1, -1):
            if bucket[index].open_time == candle.open_time:
                bucket[index] = candle
                sorted_bucket = self._sorted_candles(list(bucket))
                bucket.clear()
                bucket.extend(sorted_bucket[-self.settings.max_candle_limit :])
                return
        bucket.append(candle)
        sorted_bucket = self._sorted_candles(list(bucket))
        bucket.clear()
        bucket.extend(sorted_bucket[-self.settings.max_candle_limit :])

    def get_watchlist(self) -> list[MarketSnapshot]:
        return [self.snapshots[symbol] for symbol in self.settings.symbols if symbol in self.snapshots]

    def get_candles(self, symbol: str, interval: str, limit: int = 120) -> list[CandlePoint]:
        bucket = self.candles.get((symbol, interval))
        if not bucket:
            return []
        if limit > len(bucket) and self.settings.enable_binance_rest:
            try:
                rows = self._fetch_recent_klines(symbol, interval, limit)
                return self._sorted_candles([self._row_to_candle(symbol, interval, row) for row in rows])[-limit:]
            except Exception:
                pass
        return self._sorted_candles(list(bucket))[-limit:]

    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        *,
        from_seconds: int,
        to_seconds: int,
    ) -> list[CandlePoint]:
        from_ms = max(int(from_seconds) * 1000, 0)
        to_ms = max(int(to_seconds) * 1000, from_ms)
        if to_ms <= from_ms:
            return []

        cached = self._get_cached_history(
            symbol=symbol,
            interval=interval,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
        )
        if cached is not None:
            return cached

        if not self.settings.enable_binance_rest:
            bucket = self.candles.get((symbol, interval), [])
            candles = [
                candle
                for candle in bucket
                if from_ms <= int(datetime.fromisoformat(candle.open_time).timestamp() * 1000) < to_ms
            ]
            return self._store_cached_history(
                symbol=symbol,
                interval=interval,
                from_seconds=from_seconds,
                to_seconds=to_seconds,
                candles=candles,
            )

        step_ms = max(int(interval_to_timedelta(interval).total_seconds() * 1000), 60_000)
        rows: list[list] = []
        cursor = from_ms

        while cursor < to_ms:
            batch = self._fetch_klines_window(
                symbol,
                interval,
                start_time_ms=cursor,
                end_time_ms=to_ms,
                limit=1000,
            )
            if not batch:
                break
            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            next_cursor = last_open_ms + step_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break

        deduped = {int(row[0]): row for row in rows if from_ms <= int(row[0]) < to_ms}
        candles = [self._row_to_candle(symbol, interval, row) for _, row in sorted(deduped.items())]
        return self._store_cached_history(
            symbol=symbol,
            interval=interval,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
            candles=candles,
        )

    def get_status(self) -> LiveStatus:
        return self.live_status
