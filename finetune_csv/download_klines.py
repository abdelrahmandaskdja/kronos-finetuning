#!/usr/bin/env python3
"""
Download Binance klines and save them in Kronos CSV format.

Example:
python finetune_csv/download_klines.py \
  --symbol BTCUSDT \
  --months 6 \
  --output finetune_csv/data/BTCUSDT_kline_5min_last6m.csv

Or use exact UTC dates:
python finetune_csv/download_klines.py \
  --symbol BTCUSDT \
  --interval 1m \
  --start 2017-01-01 \
  --end 2026-01-01 \
  --output finetune_csv/data/BTCUSDT_kline_1m_2017_2025.csv
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def subtract_months(dt: datetime, months: int) -> datetime:
    """Return dt shifted back by `months` while keeping day in valid range."""
    if months <= 0:
        return dt
    year = dt.year
    month = dt.month - months
    while month <= 0:
        month += 12
        year -= 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return dt.replace(year=year, month=month, day=day)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_utc_datetime(value: str) -> datetime:
    """Parse a datetime string and return timezone-aware UTC datetime.

    Accepted formats:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM
    - YYYY-MM-DD HH:MM:SS
    - YYYY-MM-DDTHH:MM
    - YYYY-MM-DDTHH:MM:SS
    """
    value = value.strip()
    if not value:
        raise ValueError("Empty datetime string.")

    normalized = value.replace("T", " ")
    patterns = ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S")
    for pattern in patterns:
        try:
            dt = datetime.strptime(normalized, pattern)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid datetime format: {value!r}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS] (UTC)."
    )


def fetch_klines_batch(
    endpoint: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    timeout_sec: int = 20,
) -> List[list]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url=url,
        headers={"User-Agent": "kronos-klines-downloader/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {url}")
        payload = resp.read().decode("utf-8")
        data = json.loads(payload)
    if isinstance(data, dict) and "msg" in data:
        raise RuntimeError(f"API error: {data}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response format: {type(data)}")
    return data


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_sec: float = 0.1,
) -> List[list]:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    step_ms = INTERVAL_TO_MS[interval]

    current_start = start_ms
    all_rows: List[list] = []
    endpoint_errors = []
    endpoint_used = None

    for endpoint in BINANCE_ENDPOINTS:
        try:
            all_rows.clear()
            current_start = start_ms
            while current_start < end_ms:
                rows = fetch_klines_batch(
                    endpoint=endpoint,
                    symbol=symbol,
                    interval=interval,
                    start_ms=current_start,
                    end_ms=end_ms,
                )
                if not rows:
                    break

                all_rows.extend(rows)
                last_open_time = int(rows[-1][0])
                next_start = last_open_time + step_ms
                if next_start <= current_start:
                    break
                current_start = next_start
                time.sleep(pause_sec)
            endpoint_used = endpoint
            break
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as exc:
            endpoint_errors.append(f"{endpoint}: {exc}")
            continue

    if endpoint_used is None:
        err = "\n".join(endpoint_errors) if endpoint_errors else "No endpoint tried."
        raise RuntimeError(f"Failed to download klines from all endpoints.\n{err}")

    # Deduplicate by open_time to avoid overlap from pagination boundaries.
    dedup = {}
    for row in all_rows:
        dedup[int(row[0])] = row

    sorted_rows = [dedup[key] for key in sorted(dedup.keys())]
    print(f"Fetched {len(sorted_rows)} rows from {endpoint_used}")
    return sorted_rows


def candle_direction(open_price: str, close_price: str) -> int:
    # 1 = bullish/up candle, 0 = bearish/down candle.
    return 1 if float(close_price) >= float(open_price) else 0


def to_kronos_rows(rows: Iterable[list], include_direction: bool = False) -> List[list]:
    out_rows: List[list] = []
    for row in rows:
        # Binance row schema:
        # [0]=open_time, [1]=open, [2]=high, [3]=low, [4]=close,
        # [5]=volume, [7]=quote_asset_volume
        ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).strftime("%Y/%m/%d %H:%M")
        record = [
            ts,
            row[1],  # open
            row[4],  # close
            row[2],  # high
            row[3],  # low
            row[5],  # volume (base asset)
            row[7],  # amount (quote asset volume)
        ]
        if include_direction:
            record.append(candle_direction(row[1], row[4]))
        out_rows.append(record)
    return out_rows


def write_csv(path: Path, rows: List[list], include_direction: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["timestamps", "open", "close", "high", "low", "volume", "amount"]
        if include_direction:
            header.append("direction")
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance klines in Kronos CSV format")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=sorted(INTERVAL_TO_MS.keys()),
        help="Kline interval. Default is 5m.",
    )
    parser.add_argument("--months", type=int, default=6, help="How many months back from now (UTC).")
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help="Optional UTC start datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="",
        help="Optional UTC end datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetune_csv/data/BTCUSDT_kline_5min_last6m.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-direction",
        action="store_true",
        help="Append direction column (1/0) where 1 means close >= open, else 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_explicit_range = bool(args.start.strip()) or bool(args.end.strip())
    if use_explicit_range:
        if not args.start.strip() or not args.end.strip():
            raise ValueError("When using explicit range, both --start and --end are required.")
        start_dt = parse_utc_datetime(args.start)
        end_dt = parse_utc_datetime(args.end)
        if end_dt <= start_dt:
            raise ValueError("--end must be later than --start.")
    else:
        if args.months <= 0:
            raise ValueError("--months must be > 0")
        end_dt = datetime.now(timezone.utc)
        start_dt = subtract_months(end_dt, args.months)

    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)

    print(
        f"Downloading {args.symbol.upper()} {args.interval} klines "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()} (UTC)"
    )
    raw_rows = fetch_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    kronos_rows = to_kronos_rows(raw_rows, include_direction=args.include_direction)

    out_path = Path(args.output).resolve()
    write_csv(out_path, kronos_rows, include_direction=args.include_direction)
    print(f"Saved: {out_path}")
    if kronos_rows:
        print(f"Range: {kronos_rows[0][0]} -> {kronos_rows[-1][0]} (UTC)")
    print(f"Total rows: {len(kronos_rows)}")


if __name__ == "__main__":
    main()
