#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer
from finetune_csv.download_klines import INTERVAL_TO_MS, fetch_klines, to_kronos_rows, to_ms, write_csv
from finetune_csv.eval_consecutive_rollout import build_stability_profile, load_csv as load_rollout_csv, stabilize_pred_block
from finetune_csv.eval_direction_model import load_checkpoint_model


LIVE_SPOT_BASE_URL = "https://api.binance.com"
TESTNET_SPOT_BASE_URL = "https://testnet.binance.vision"
DEMO_SPOT_BASE_URL = "https://demo-api.binance.com"
LIVE_SPOT_STREAM_BASE_URL = "wss://stream.binance.com:443"
TESTNET_SPOT_STREAM_BASE_URL = "wss://stream.testnet.binance.vision"
LIVE_SPOT_WS_API_URL = "wss://ws-api.binance.com:443/ws-api/v3"
TESTNET_SPOT_WS_API_URL = "wss://ws-api.testnet.binance.vision/ws-api/v3"

_EXCHANGE_TIME_LOCK = threading.Lock()
_EXCHANGE_TIME_OFFSET_MS = 0
_EXCHANGE_TIME_LAST_SYNC_LOCAL_MS = 0


@dataclass(frozen=True)
class ModelSignal:
    model_id: str
    model_name: str
    family: str
    role: str
    metric_name: str
    interval: str
    binance_interval: str
    context_timestamp: str
    target_timestamp: str
    direction: str
    signal_value: float
    score: float
    confidence_pct: float | None
    move_pct: float | None
    prob_up: float | None
    reference_price: float | None
    predicted_price: float | None
    recent_vol_pct: float | None
    market_path: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_ms() -> int:
    return int(utc_now().timestamp() * 1000)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def update_exchange_time_offset(offset_ms: int, synced_local_ms: int | None = None) -> int:
    global _EXCHANGE_TIME_OFFSET_MS, _EXCHANGE_TIME_LAST_SYNC_LOCAL_MS
    with _EXCHANGE_TIME_LOCK:
        _EXCHANGE_TIME_OFFSET_MS = int(offset_ms)
        if synced_local_ms is not None:
            _EXCHANGE_TIME_LAST_SYNC_LOCAL_MS = int(synced_local_ms)
        return _EXCHANGE_TIME_OFFSET_MS


def current_exchange_time_offset_ms() -> int:
    with _EXCHANGE_TIME_LOCK:
        return _EXCHANGE_TIME_OFFSET_MS


def ensure_exchange_time_offset(base_url: str, max_age_ms: int = 300_000) -> int:
    with _EXCHANGE_TIME_LOCK:
        last_sync_local_ms = _EXCHANGE_TIME_LAST_SYNC_LOCAL_MS
        current_offset_ms = _EXCHANGE_TIME_OFFSET_MS
    local_now_ms = utc_now_ms()
    if last_sync_local_ms and abs(local_now_ms - last_sync_local_ms) <= max(int(max_age_ms), 1):
        return current_offset_ms
    server_time_ms = fetch_exchange_server_time_ms(base_url)
    offset_ms = int(server_time_ms) - int(local_now_ms)
    return update_exchange_time_offset(offset_ms, synced_local_ms=local_now_ms)


def exchange_utc_now() -> datetime:
    return datetime.fromtimestamp(exchange_utc_now_ms() / 1000, tz=timezone.utc)


def exchange_utc_now_ms() -> int:
    return utc_now_ms() + current_exchange_time_offset_ms()


def exchange_utc_now_iso() -> str:
    return exchange_utc_now().isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def load_env_file(env_path: Path | None) -> None:
    if env_path is None or not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # An explicit env file should pin the bot to that interval's account.
        os.environ[key] = value


def resolve_optional_runtime_path(config: dict[str, Any], key: str, default_filename: str) -> Path:
    raw_path = str(config.get("paths", {}).get(key, "")).strip()
    if raw_path:
        return Path(raw_path)
    runtime_root = Path(config["paths"]["runtime_root"])
    return runtime_root / default_filename


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def infer_time_delta(timestamps: pd.Series, fallback: pd.Timedelta | None = None) -> pd.Timedelta | None:
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return fallback
    try:
        return diffs.mode().iloc[0]
    except (IndexError, ValueError):
        return diffs.iloc[-1]


def load_market_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamps" not in df.columns:
        raise ValueError(f"Missing 'timestamps' column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamps"], errors="coerce", utc=False)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    return df


def build_prediction_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        columns.append("volume")
    if "amount" in df.columns:
        columns.append("amount")
    return df[columns].copy()


def write_market_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "timestamp" in out.columns and "timestamps" not in out.columns:
        out["timestamps"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y/%m/%d %H:%M")
    elif "timestamps" in out.columns:
        out["timestamps"] = pd.to_datetime(out["timestamps"]).dt.strftime("%Y/%m/%d %H:%M")
    else:
        raise ValueError("Market dataframe must contain 'timestamp' or 'timestamps'")
    ordered = ["timestamps", "open", "close", "high", "low", "volume", "amount"]
    missing = [column for column in ordered if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")
    out[ordered].to_csv(path, index=False)


def sanitize_forecast_frame(df: pd.DataFrame, allow_non_positive: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    if not allow_non_positive and (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Forecast produced non-positive prices")
    return out


def apply_consecutive_direction_rule(direction_mode: str, forecast_df: pd.DataFrame, last_close: float) -> pd.DataFrame:
    out = forecast_df.copy()
    if direction_mode == "close_to_close":
        reference_close = float(last_close)
        ref_values: list[float] = []
        pred_dirs: list[int] = []
        move_pcts: list[float | None] = []
        for _, row in out.iterrows():
            ref_values.append(reference_close)
            pred_close = float(row["close"])
            pred_dirs.append(1 if pred_close >= reference_close else 0)
            if reference_close == 0:
                move_pcts.append(None)
            else:
                move_pcts.append(((pred_close / reference_close) - 1.0) * 100.0)
            reference_close = pred_close
        out["reference_close"] = ref_values
        out["pred_dir"] = pred_dirs
        out["move_pct"] = move_pcts
        return out

    base_open = out["open"].replace(0, np.nan)
    out["pred_dir"] = (out["close"] >= out["open"]).astype(int)
    out["move_pct"] = ((out["close"] / base_open) - 1.0) * 100.0
    return out


def decimal_from(value: Any) -> Decimal:
    return Decimal(str(value))


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def quantize_up(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


@lru_cache(maxsize=16)
def load_stability_profile(context_path: str):
    args = SimpleNamespace(
        stability_quantile=0.995,
        stability_multiplier=6.0,
        min_return_cap=0.02,
        max_return_cap=1.0,
    )
    train_df = load_rollout_csv(Path(context_path).expanduser().resolve())
    return build_stability_profile(train_df, args)


def load_best_local_market_fallback(fallback_paths: list[str], rows_needed: int) -> pd.DataFrame | None:
    best_df: pd.DataFrame | None = None
    best_end_ts: pd.Timestamp | None = None
    for raw_path in fallback_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        try:
            df = load_market_dataframe(path)
        except Exception:
            continue
        if len(df) < rows_needed:
            continue
        end_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        if best_df is None or end_ts > best_end_ts:
            best_df = df
            best_end_ts = end_ts
    return best_df


def fetch_public_binance_json(base_url: str, path: str, params: dict[str, Any] | None = None, timeout_seconds: int = 20) -> Any:
    query = urllib.parse.urlencode({key: value for key, value in (params or {}).items() if value is not None}, doseq=True)
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": "kronos-live-market-fetch/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def fetch_exchange_server_time_ms(base_url: str) -> int:
    payload = fetch_public_binance_json(base_url, "/api/v3/time")
    return int(payload["serverTime"])


def fetch_klines_from_base_url(
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_sec: float = 0.05,
    limit: int = 1000,
) -> list[list]:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    step_ms = INTERVAL_TO_MS[interval]
    current_start = start_ms
    all_rows: list[list] = []

    while current_start < end_ms:
        rows = fetch_public_binance_json(
            base_url=base_url,
            path="/api/v3/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected kline response format from {base_url}: {type(rows)}")
        if not rows:
            break
        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        next_start = last_open_time + step_ms
        if next_start <= current_start:
            break
        current_start = next_start
        time.sleep(pause_sec)

    dedup = {int(row[0]): row for row in all_rows}
    sorted_rows = [dedup[key] for key in sorted(dedup.keys())]
    print(f"Fetched {len(sorted_rows)} rows from {base_url.rstrip('/')}/api/v3/klines")
    return sorted_rows


def build_default_config() -> dict[str, Any]:
    cross_tf_2026_data_root = (
        REPO_ROOT
        / "reports"
        / "ad_hoc_20260409"
        / "cross_timeframe_1m_trainall_6epoch_2026-04-08"
        / "data"
    )
    biasctrl_root = (
        REPO_ROOT
        / "finetune_csv"
        / "runs_requested_by_user"
        / "research_1m_trainall_2025H1_biasctrl_20260406_221436"
        / "finetuned"
        / "BTCUSDT_kline_1m_trainall_2025H1_biasctrl"
    )
    leakfix_root = (
        REPO_ROOT
        / "finetune_csv"
        / "runs_requested_by_user"
        / "retrain_1m_2017_2024_ce_consdir_leakfix_20260409_214258"
        / "finetuned"
        / "BTCUSDT_kline_1m_2017_2024_ce_consdir_leakfix"
    )
    direction_head_ckpt = (
        REPO_ROOT
        / "finetune_csv"
        / "runs_requested_by_user"
        / "research_5min_direction_head_oldsplit_20260318"
        / "checkpoints"
        / "best_direction_model.pt"
    )
    runtime_root = REPO_ROOT / "trading" / "runtime"
    return {
        "symbol": "BTCUSDT",
        "market": "spot",
        "mode": "dry_run",
        "api": {
            "key_env": "BINANCE_API_KEY",
            "secret_env": "BINANCE_API_SECRET",
            "recv_window_ms": 5000,
            "timeout_seconds": 20,
        },
        "paths": {
            "runtime_root": str(runtime_root),
            "state_path": str(runtime_root / "state.json"),
            "signal_log_path": str(runtime_root / "signal_log.jsonl"),
            "order_log_path": str(runtime_root / "order_log.jsonl"),
            "status_log_path": str(runtime_root / "status_log.jsonl"),
            "market_cache_dir": str(runtime_root / "market_cache"),
        },
        "strategy": {
            "mode": "ensemble",
            "single_interval": "",
            "primary_interval": "15m",
            "primary_interval_minutes": 15,
            "max_target_allocation": 1.0,
            "probability_edge_for_full_score": 0.10,
            "sync_vote_threshold": 0.55,
            "allocation_ladder": {
                "base_when_regime_and_entry_align": 0.50,
                "one_hour_bonus": 0.25,
                "one_minute_bonus": 0.15,
                "strong_vote_bonus": 0.10,
            },
        },
        "risk": {
            "long_only": True,
            "flatten_on_regime_flip": True,
            "min_rebalance_notional_usdt": 25.0,
            "max_trade_notional_usdt": 250.0,
            "max_daily_drawdown_pct": 6.0,
            "max_intraday_drawdown_pct": 4.0,
            "default_paper_equity_usdt": 1000.0,
        },
        "execution": {
            "order_style": "maker_limit",
            "maker_client_order_prefix": "kronosmkr",
            "maker_inside_spread_ticks": 1,
            "maker_reprice_after_ticks": 1,
            "maker_buy_balance_buffer_pct": 0.25,
            "maker_cancel_on_idle": True,
            "maker_cancel_on_signal_flip": True,
            "allow_same_bar_order_maintenance": True,
            "same_bar_maintenance_requires_existing_order": False,
            "primary_close_buffer_ms": 250,
            "primary_same_bar_maintenance_interval_ms": 5000,
            "ws_order_entry": False,
            "use_primary_peg_limit_maker": False,
            "maker_peg_offset_levels": 0,
            "fast_1m_enabled": False,
            "fast_1m_close_buffer_ms": 150,
            "fast_1m_same_bar_maintenance_interval_ms": 500,
            "fast_1m_ws_orders": False,
            "fast_1m_inference_device": "auto",
            "fast_1m_prewarm": True,
            "fast_1m_use_full_forecast_cycle": False,
        },
        "models": {
            "1d": {
                "model_id": "1d_leakfix",
                "model_name": "BTCUSDT_kline_1m_2017_2024_ce_consdir_leakfix",
                "family": "consecutive_rollout",
                "role": "regime",
                "metric_name": "pred_close_vs_prev_actual_close",
                "interval": "1d",
                "binance_interval": "1d",
                "model_path": str(leakfix_root / "basemodel" / "best_model"),
                "tokenizer_path": str(leakfix_root / "tokenizer" / "best_model"),
                "stability_context_path": str(REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_1d_2017_2025.csv"),
                "fallback_market_paths": [
                    str(cross_tf_2026_data_root / "BTCUSDT_kline_1d_2026-01-01_to_2026-04-08_eval.csv"),
                    str(REPO_ROOT / "finetune_csv" / "data" / "live_binance" / "BTCUSDT_kline_1day_live_binance.csv"),
                    str(REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_1d_2026_to_now.csv"),
                ],
                "stabilize_output": True,
                "lookback": 256,
                "pred_len": 16,
                "use_full_forecast_cycle": True,
                "clip": 5.0,
                "device": "cpu",
                "direction_mode": "close_to_close",
                "combined_pnl_usdt": 24261.33,
            },
            "4h": {
                "model_id": "4h_biasctrl_epoch6_lowest_ce",
                "model_name": "BTCUSDT_kline_1m_trainall_2025H1_biasctrl_epoch6_lowest_ce",
                "family": "consecutive_rollout",
                "role": "regime",
                "metric_name": "pred_close_vs_prev_actual_close",
                "interval": "4h",
                "binance_interval": "4h",
                "model_path": str(biasctrl_root / "_rollout_checkpoint_tmp"),
                "tokenizer_path": str(biasctrl_root / "tokenizer" / "best_model"),
                "stability_context_path": str(
                    REPO_ROOT
                    / "reports"
                    / "ad_hoc_20260411"
                    / "interval_winners_full_2025"
                    / "data"
                    / "BTCUSDT_kline_4h_2024-09-01_to_2025-12-31_exact.csv"
                ),
                "fallback_market_paths": [
                    str(cross_tf_2026_data_root / "BTCUSDT_kline_4h_2026-01-01_to_2026-04-08_eval.csv"),
                ],
                "stabilize_output": True,
                "lookback": 256,
                "pred_len": 16,
                "use_full_forecast_cycle": True,
                "clip": 5.0,
                "device": "cpu",
                "direction_mode": "close_to_close",
                "combined_pnl_usdt": 46691.67,
            },
            "1h": {
                "model_id": "1h_biasctrl_best",
                "model_name": "BTCUSDT_kline_1m_trainall_2025H1_biasctrl",
                "family": "consecutive_rollout",
                "role": "confirm",
                "metric_name": "pred_close_vs_prev_actual_close",
                "interval": "1h",
                "binance_interval": "1h",
                "model_path": str(biasctrl_root / "basemodel" / "best_model"),
                "tokenizer_path": str(biasctrl_root / "tokenizer" / "best_model"),
                "stability_context_path": str(REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_1h_2018_2025.csv"),
                "fallback_market_paths": [
                    str(cross_tf_2026_data_root / "BTCUSDT_kline_1h_2026-01-01_to_2026-04-08_eval.csv"),
                    str(REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_1h_2026_to_now.csv"),
                ],
                "stabilize_output": True,
                "lookback": 256,
                "pred_len": 16,
                "use_full_forecast_cycle": True,
                "clip": 5.0,
                "device": "cpu",
                "direction_mode": "close_to_close",
                "combined_pnl_usdt": 10695.15,
            },
            "15m": {
                "model_id": "15m_5min_native_direction_head",
                "model_name": "5min native direction head",
                "family": "direction_head",
                "role": "entry",
                "metric_name": "pred_dir_vs_next_bar_label",
                "interval": "15m",
                "binance_interval": "15m",
                "checkpoint_path": str(direction_head_ckpt),
                "lookback": 512,
                "fallback_market_paths": [
                    str(cross_tf_2026_data_root / "BTCUSDT_kline_15m_2026-01-01_to_2026-04-08_eval.csv"),
                    str(REPO_ROOT / "finetune_csv" / "data" / "live_binance" / "BTCUSDT_kline_15min_live_binance.csv"),
                    str(
                        REPO_ROOT
                        / "finetune_csv"
                        / "runs_requested_by_user"
                        / "retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313"
                        / "eval_epoch1_2026_to_2026-03-21_data"
                        / "BTCUSDT_kline_15m_2026_to_2026-03-21_live.csv"
                    ),
                ],
                "device": "cpu",
                "combined_pnl_usdt": 166542.11,
            },
            "1m": {
                "model_id": "1m_leakfix",
                "model_name": "BTCUSDT_kline_1m_2017_2024_ce_consdir_leakfix",
                "family": "consecutive_rollout",
                "role": "micro",
                "metric_name": "pred_close_vs_prev_actual_close",
                "interval": "1m",
                "binance_interval": "1m",
                "model_path": str(leakfix_root / "basemodel" / "best_model"),
                "tokenizer_path": str(leakfix_root / "tokenizer" / "best_model"),
                "stability_context_path": str(REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_1m_2017_2026_full.csv"),
                "fallback_market_paths": [
                    str(cross_tf_2026_data_root / "BTCUSDT_kline_1m_2026-01-01_to_2026-04-08_eval.csv"),
                    str(
                        REPO_ROOT
                        / "finetune_csv"
                        / "runs_requested_by_user"
                        / "research_1m_trainall_2025H1_biasctrl_20260406_221436"
                        / "data"
                        / "BTCUSDT_kline_1m_2026_to_now_test.csv"
                    ),
                    str(
                        REPO_ROOT
                        / "finetune_csv"
                        / "runs_requested_by_user"
                        / "research_1m_20260310_run1"
                        / "data"
                        / "BTCUSDT_kline_1m_2026_test.csv"
                    ),
                ],
                "stabilize_output": True,
                "lookback": 256,
                "pred_len": 16,
                "use_full_forecast_cycle": True,
                "clip": 5.0,
                "device": "cpu",
                "direction_mode": "close_to_close",
                "combined_pnl_usdt": 18149.79,
            },
        },
    }


@lru_cache(maxsize=8)
def load_consecutive_predictor(model_path: str, tokenizer_path: str, max_context: int, clip: float, device: str) -> KronosPredictor:
    device = resolve_runtime_device(device)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context, clip=clip)


@lru_cache(maxsize=8)
def load_direction_head_bundle(checkpoint_path: str, device: str):
    resolved_device = torch.device(resolve_runtime_device(device))
    ckpt, model, _, _ = load_checkpoint_model(
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        device=resolved_device,
        predictor_path_override=None,
    )
    cfg = ckpt.get("config", {})
    tokenizer_path = ckpt.get("tokenizer_path") or cfg.get("pretrained_tokenizer_path")
    if not tokenizer_path:
        raise ValueError(f"Could not resolve tokenizer path for {checkpoint_path}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.to(resolved_device)
    tokenizer.eval()
    model.eval()
    return {
        "device": resolved_device,
        "checkpoint": ckpt,
        "config": cfg,
        "model": model,
        "tokenizer": tokenizer,
        "tokenizer_path": tokenizer_path,
    }


def fetch_latest_closed_klines(
    base_url: str,
    symbol: str,
    interval: str,
    rows_needed: int,
    output_path: Path,
    fallback_paths: list[str] | None = None,
) -> Path:
    interval_ms = INTERVAL_TO_MS[interval]
    fetch_error: Exception | None = None
    try:
        end_ms = fetch_exchange_server_time_ms(base_url)
    except Exception as exc:
        end_dt = utc_now()
        end_ms = to_ms(end_dt)
        fetch_error = exc
    # Start slightly earlier than the nominal lookback so dropping the still-open bar
    # does not leave the cache one row short on larger intervals such as 1d.
    buffer_rows = max(2, min(8, rows_needed // 32 or 1))
    start_ms = end_ms - ((rows_needed + buffer_rows) * interval_ms)
    try:
        rows = fetch_klines_from_base_url(
            base_url=base_url,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            pause_sec=0.05,
        )
    except Exception as exc:
        try:
            rows = fetch_klines(
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                pause_sec=0.05,
            )
        except Exception as fallback_exc:
            rows = []
            fetch_error = fallback_exc if fetch_error is None else RuntimeError(f"{fetch_error}; market fetch failed: {fallback_exc}")
    if rows and int(rows[-1][0]) + interval_ms > end_ms:
        rows = rows[:-1]
    if len(rows) >= rows_needed:
        rows_to_write = rows[-rows_needed:]
        write_csv(output_path, to_kronos_rows(rows_to_write), include_direction=False)
        return output_path

    fallback_df = load_best_local_market_fallback(fallback_paths or [], rows_needed)
    if fallback_df is not None:
        write_market_dataframe(output_path, fallback_df.tail(rows_needed))
        return output_path

    if not rows:
        if fetch_error is not None:
            raise RuntimeError(f"Failed to fetch rows for {symbol} {interval}: {fetch_error}")
        raise RuntimeError(f"No closed rows fetched for {symbol} {interval}")
    raise RuntimeError(f"Only fetched {len(rows)} closed rows for {symbol} {interval}; need {rows_needed}")


def compute_recent_vol_pct(market_df: pd.DataFrame, vol_floor_pct: float, tail_window: int = 96) -> float:
    returns = market_df["close"].pct_change().dropna() * 100.0
    if returns.empty:
        return float(vol_floor_pct)
    recent = returns.tail(min(tail_window, len(returns)))
    std_value = float(recent.std(ddof=0)) if len(recent) > 1 else float(abs(recent.iloc[-1]))
    return max(std_value, float(vol_floor_pct))


def stabilize_live_consecutive_forecast(spec: dict[str, Any], context_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    if not spec.get("stabilize_output", True):
        return forecast_df
    stability_context_path = spec.get("stability_context_path")
    if not stability_context_path:
        return forecast_df
    profile = load_stability_profile(str(stability_context_path))
    pred_block = forecast_df.rename(columns={"timestamp": "timestamps"})[
        ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    ].copy()
    stable_block, _ = stabilize_pred_block(
        pred_block=pred_block,
        prev_close_seed=float(context_df["close"].iloc[-1]),
        profile=profile,
    )
    return sanitize_forecast_frame(
        stable_block.rename(columns={"timestamps": "timestamp"}),
        allow_non_positive=True,
    )


def generate_consecutive_forecast(spec: dict[str, Any], market_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_df = load_market_dataframe(market_path)
    if len(market_df) < int(spec["lookback"]):
        raise ValueError(f"Not enough rows for {spec['model_id']}: need {spec['lookback']}, got {len(market_df)}")
    context_df = market_df.tail(int(spec["lookback"])).copy()
    fallback_delta = pd.Timedelta(milliseconds=INTERVAL_TO_MS[str(spec["binance_interval"])])
    time_delta = infer_time_delta(context_df["timestamp"], fallback=fallback_delta)
    if time_delta is None:
        raise ValueError(f"Could not infer bar spacing for {spec['model_id']}")
    y_timestamp = pd.Series(
        pd.date_range(
            start=context_df["timestamp"].iloc[-1] + time_delta,
            periods=int(spec.get("pred_len", 1)),
            freq=time_delta,
        )
    )
    predictor = load_consecutive_predictor(
        model_path=str(spec["model_path"]),
        tokenizer_path=str(spec["tokenizer_path"]),
        max_context=int(spec["lookback"]),
        clip=float(spec.get("clip", 5.0)),
        device=str(spec.get("device", "cpu")),
    )
    pred_df = predictor.predict(
        build_prediction_input_frame(context_df),
        context_df["timestamp"],
        y_timestamp,
        pred_len=int(spec.get("pred_len", 1)),
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    forecast_df = sanitize_forecast_frame(
        pred_df.reset_index().rename(columns={"index": "timestamp"}),
        allow_non_positive=True,
    )
    forecast_df = stabilize_live_consecutive_forecast(spec, context_df, forecast_df)
    forecast_df = apply_consecutive_direction_rule(str(spec.get("direction_mode", "close_to_close")), forecast_df, float(context_df["close"].iloc[-1]))
    return context_df, forecast_df


def build_consecutive_signal_from_row(
    spec: dict[str, Any],
    context_timestamp: pd.Timestamp,
    market_path: Path,
    row: pd.Series,
) -> ModelSignal:
    move_pct = None if pd.isna(row.get("move_pct")) else float(row["move_pct"])
    signal_value = 1.0 if int(row["pred_dir"]) == 1 else -1.0
    confidence_pct = abs(move_pct) if move_pct is not None else None
    return ModelSignal(
        model_id=str(spec["model_id"]),
        model_name=str(spec["model_name"]),
        family=str(spec["family"]),
        role=str(spec["role"]),
        metric_name=str(spec["metric_name"]),
        interval=str(spec["interval"]),
        binance_interval=str(spec["binance_interval"]),
        context_timestamp=str(pd.Timestamp(context_timestamp).isoformat()),
        target_timestamp=str(pd.Timestamp(row["timestamp"]).isoformat()),
        direction="UP" if int(row["pred_dir"]) == 1 else "DOWN",
        signal_value=signal_value,
        score=signal_value,
        confidence_pct=confidence_pct,
        move_pct=move_pct,
        prob_up=None,
        reference_price=float(row.get("reference_close")),
        predicted_price=float(row["close"]),
        recent_vol_pct=None,
        market_path=str(market_path),
    )


def generate_consecutive_signal(config: dict[str, Any], spec: dict[str, Any], market_path: Path) -> ModelSignal:
    context_df, forecast_df = generate_consecutive_forecast(spec, market_path)
    metric_name = str(spec.get("metric_name", "pred_close_vs_prev_actual_close"))
    signal_row_index = 0
    if metric_name == "pred_close_vs_prev_pred_close":
        if len(forecast_df) < 2:
            raise ValueError(f"Metric {metric_name} requires pred_len >= 2 for {spec['model_id']}")
        signal_row_index = 1
    elif metric_name != "pred_close_vs_prev_actual_close":
        raise ValueError(f"Unsupported consecutive metric_name for live trading: {metric_name}")
    next_row = forecast_df.iloc[signal_row_index]
    return build_consecutive_signal_from_row(
        spec=spec,
        context_timestamp=pd.Timestamp(context_df["timestamp"].iloc[-1]),
        market_path=market_path,
        row=next_row,
    )


def generate_direction_head_signal(config: dict[str, Any], spec: dict[str, Any], market_path: Path) -> ModelSignal:
    bundle = load_direction_head_bundle(str(spec["checkpoint_path"]), str(spec.get("device", "cpu")))
    ckpt_cfg = bundle["config"]
    lookback = int(spec.get("lookback") or ckpt_cfg.get("lookback_window", 400))
    clip = float(spec.get("clip") or ckpt_cfg.get("clip", 5.0))
    horizon_steps = int(spec.get("horizon_steps") or ckpt_cfg.get("horizon_steps", 1))
    market_df = load_market_dataframe(market_path)
    if len(market_df) < lookback:
        raise ValueError(f"Not enough rows for {spec['model_id']}: need {lookback}, got {len(market_df)}")
    hist_df = market_df.tail(lookback).copy()
    feature_columns = ["open", "high", "low", "close", "volume", "amount"]
    x = hist_df[feature_columns].to_numpy(dtype=np.float32)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x = np.clip((x - x_mean) / (x_std + 1e-5), -clip, clip).astype(np.float32)
    stamp_df = pd.DataFrame(
        {
            "minute": hist_df["timestamp"].dt.minute.astype(np.float32),
            "hour": hist_df["timestamp"].dt.hour.astype(np.float32),
            "weekday": hist_df["timestamp"].dt.weekday.astype(np.float32),
            "day": hist_df["timestamp"].dt.day.astype(np.float32),
            "month": hist_df["timestamp"].dt.month.astype(np.float32),
        }
    )
    device = bundle["device"]
    x_tensor = torch.from_numpy(x[None, :, :]).to(device)
    stamp_tensor = torch.from_numpy(stamp_df.to_numpy(dtype=np.float32)[None, :, :]).to(device)
    with torch.no_grad():
        token_s1, token_s2 = bundle["tokenizer"].encode(x_tensor, half=True)
        logits = bundle["model"](token_s1, token_s2, stamp_tensor)
        prob_up = float(torch.sigmoid(logits).detach().cpu().reshape(-1)[0].item())
    centered = prob_up - 0.5
    signal_value = 1.0 if prob_up >= 0.5 else -1.0
    confidence_pct = abs(centered) * 200.0
    fallback_delta = pd.Timedelta(milliseconds=INTERVAL_TO_MS[str(spec["binance_interval"])])
    time_delta = infer_time_delta(hist_df["timestamp"], fallback=fallback_delta) or fallback_delta
    target_timestamp = pd.Timestamp(hist_df["timestamp"].iloc[-1]) + (horizon_steps * time_delta)
    return ModelSignal(
        model_id=str(spec["model_id"]),
        model_name=str(spec["model_name"]),
        family=str(spec["family"]),
        role=str(spec["role"]),
        metric_name=str(spec["metric_name"]),
        interval=str(spec["interval"]),
        binance_interval=str(spec["binance_interval"]),
        context_timestamp=str(pd.Timestamp(hist_df["timestamp"].iloc[-1]).isoformat()),
        target_timestamp=str(target_timestamp.isoformat()),
        direction="UP" if prob_up >= 0.5 else "DOWN",
        signal_value=signal_value,
        score=signal_value,
        confidence_pct=confidence_pct,
        move_pct=None,
        prob_up=prob_up,
        reference_price=float(hist_df["close"].iloc[-1]),
        predicted_price=None,
        recent_vol_pct=None,
        market_path=str(market_path),
    )


def build_market_path(config: dict[str, Any], spec: dict[str, Any]) -> Path:
    market_cache_dir = Path(config["paths"]["market_cache_dir"])
    market_cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{config['symbol']}_{spec['binance_interval']}.csv"
    return market_cache_dir / filename


def get_strategy_mode(config: dict[str, Any]) -> str:
    return str(config.get("strategy", {}).get("mode", "ensemble")).lower()


def get_active_model_keys(config: dict[str, Any]) -> list[str]:
    mode = get_strategy_mode(config)
    all_keys = list(config["models"].keys())
    if mode != "single_interval":
        return all_keys
    single_interval = str(config.get("strategy", {}).get("single_interval", "")).strip()
    if not single_interval:
        raise ValueError("strategy.single_interval must be set when strategy.mode=single_interval")
    if single_interval not in config["models"]:
        raise ValueError(f"Unknown single_interval strategy target: {single_interval}")
    return [single_interval]


def build_signal_map(
    config: dict[str, Any],
    base_url: str,
    state: dict[str, Any],
) -> tuple[dict[str, ModelSignal], dict[str, str], dict[str, Any]]:
    signals: dict[str, ModelSignal] = {}
    errors: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    for interval_key in get_active_model_keys(config):
        spec = config["models"][interval_key]
        try:
            if spec["family"] == "direction_head":
                bundle = load_direction_head_bundle(str(spec["checkpoint_path"]), str(spec.get("device", "cpu")))
                rows_needed = int(spec.get("lookback") or bundle["config"].get("lookback_window", 400)) + 4
            else:
                rows_needed = int(spec.get("lookback", 256)) + int(spec.get("pred_len", 1)) + 4
            market_path = build_market_path(config, spec)
            fetch_latest_closed_klines(
                base_url=base_url,
                symbol=str(config["symbol"]),
                interval=str(spec["binance_interval"]),
                rows_needed=rows_needed,
                output_path=market_path,
                fallback_paths=list(spec.get("fallback_market_paths", [])),
            )
            if spec["family"] == "direction_head":
                signals[interval_key] = generate_direction_head_signal(config, spec, market_path)
            elif is_full_consecutive_forecast_cycle_enabled(spec):
                state_key = f"forecast_cycle_{interval_key}"
                signals[interval_key], diagnostics[interval_key] = resolve_interval_forecast_cycle_signal(
                    state=state,
                    state_key=state_key,
                    spec=spec,
                    market_path=market_path,
                )
            else:
                signals[interval_key] = generate_consecutive_signal(config, spec, market_path)
        except Exception as exc:
            errors[interval_key] = f"{spec['model_id']}: {exc}"
    return signals, errors, diagnostics


def compute_model_weights(config: dict[str, Any]) -> dict[str, float]:
    raw_values: dict[str, float] = {}
    for interval_key, spec in config["models"].items():
        combined_pnl = max(float(spec.get("combined_pnl_usdt", 1.0)), 1.0)
        raw_values[interval_key] = math.log1p(combined_pnl)
    total = sum(raw_values.values())
    return {key: value / total for key, value in raw_values.items()}


def build_single_interval_strategy(config: dict[str, Any], signals: dict[str, ModelSignal]) -> dict[str, Any]:
    interval_key = str(config["strategy"]["single_interval"])
    signal = signals[interval_key]
    signed_votes = {interval_key: float(signal.signal_value)}
    signed_vote = float(signal.signal_value)
    max_alloc = float(config["strategy"]["max_target_allocation"])
    target_exposure = signed_vote * max_alloc if signed_vote != 0 else 0.0
    reasons: list[str] = []
    if signed_vote < 0:
        reasons.append(f"{interval_key} standalone signal is bearish")
    elif signed_vote > 0:
        reasons.append(f"{interval_key} standalone signal is bullish")
    else:
        reasons.append(f"{interval_key} standalone signal is flat")

    if config["risk"].get("long_only", True):
        execution_target_allocation = max(target_exposure, 0.0)
    else:
        execution_target_allocation = target_exposure

    bullish_intervals = [interval_key] if signed_vote > 0 else []
    bearish_intervals = [interval_key] if signed_vote < 0 else []
    return {
        "weights": {interval_key: 1.0},
        "metric_names": {interval_key: signal.metric_name},
        "signed_votes": signed_votes,
        "signed_vote": signed_vote,
        "aligned_side": signed_vote,
        "target_exposure": target_exposure,
        "target_allocation": execution_target_allocation,
        "bullish_intervals": bullish_intervals,
        "bearish_intervals": bearish_intervals,
        "reasons": reasons,
        "regime_ok": True,
        "entry_ok": signed_vote > 0,
        "strategy_mode": "single_interval",
        "single_interval": interval_key,
    }


def build_synchronized_strategy(config: dict[str, Any], signals: dict[str, ModelSignal]) -> dict[str, Any]:
    if get_strategy_mode(config) == "single_interval":
        return build_single_interval_strategy(config, signals)

    weights = compute_model_weights(config)
    signed_votes = {key: float(signals[key].signal_value) for key in signals}
    signed_vote = sum(weights[key] * signed_votes[key] for key in signals)
    regime_daily = signed_votes["1d"]
    regime_4h = signed_votes["4h"]
    entry_15m = signed_votes["15m"]
    confirm_1h = signed_votes["1h"]
    micro_1m = signed_votes["1m"]

    base_alloc = float(config["strategy"]["allocation_ladder"]["base_when_regime_and_entry_align"])
    bonus_1h = float(config["strategy"]["allocation_ladder"]["one_hour_bonus"])
    bonus_1m = float(config["strategy"]["allocation_ladder"]["one_minute_bonus"])
    bonus_vote = float(config["strategy"]["allocation_ladder"]["strong_vote_bonus"])
    max_alloc = float(config["strategy"]["max_target_allocation"])
    sync_vote_threshold = float(config["strategy"]["sync_vote_threshold"])

    reasons: list[str] = []
    target_exposure = 0.0
    aligned_side = 0.0

    if regime_daily != regime_4h:
        reasons.append("1d and 4h regime signals disagree")
    if regime_daily != entry_15m:
        reasons.append("15m entry signal is not aligned with the higher-timeframe regime")

    if not reasons:
        aligned_side = entry_15m
        target_abs = base_alloc
        if confirm_1h == aligned_side:
            target_abs += bonus_1h
        if micro_1m == aligned_side:
            target_abs += bonus_1m
        if np.sign(signed_vote) == np.sign(aligned_side) and abs(signed_vote) >= sync_vote_threshold:
            target_abs += bonus_vote
        target_exposure = float(np.clip(aligned_side * min(target_abs, max_alloc), -max_alloc, max_alloc))

    if config["risk"].get("long_only", True):
        execution_target_allocation = max(target_exposure, 0.0)
    else:
        execution_target_allocation = target_exposure

    bullish_intervals = [key for key, signal in signals.items() if signal.signal_value > 0]
    bearish_intervals = [key for key, signal in signals.items() if signal.signal_value < 0]
    return {
        "weights": weights,
        "metric_names": {key: signals[key].metric_name for key in signals},
        "signed_votes": signed_votes,
        "signed_vote": signed_vote,
        "aligned_side": aligned_side,
        "target_exposure": target_exposure,
        "target_allocation": execution_target_allocation,
        "bullish_intervals": bullish_intervals,
        "bearish_intervals": bearish_intervals,
        "reasons": reasons,
        "regime_ok": regime_daily == regime_4h,
        "entry_ok": regime_daily == entry_15m,
        "strategy_mode": "ensemble",
    }


class BinanceSpotClient:
    def __init__(self, base_url: str, api_key: str | None, api_secret: str | None, timeout_seconds: int = 20):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.timeout_seconds = int(timeout_seconds)
        self.time_offset_ms = 0
        self.last_time_sync_ms = 0

    def get_server_time(self) -> dict[str, Any]:
        return self._request("GET", "/api/v3/time", signed=False)

    def sync_server_time(self) -> int:
        server_time_payload = self.get_server_time()
        server_time_ms = int(server_time_payload["serverTime"])
        local_time_ms = utc_now_ms()
        self.time_offset_ms = server_time_ms - local_time_ms
        self.last_time_sync_ms = local_time_ms
        update_exchange_time_offset(self.time_offset_ms, synced_local_ms=local_time_ms)
        return self.time_offset_ms

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        allow_time_resync: bool = True,
    ) -> Any:
        params = {key: value for key, value in (params or {}).items() if value is not None}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("Signed Binance request requires API key and secret")
            if self.last_time_sync_ms == 0 or abs(utc_now_ms() - self.last_time_sync_ms) > 300_000:
                self.sync_server_time()
            params["timestamp"] = utc_now_ms() + self.time_offset_ms
            query = urllib.parse.urlencode(params, doseq=True)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            query = f"{query}&signature={signature}"
            headers = {"X-MBX-APIKEY": self.api_key}
        else:
            query = urllib.parse.urlencode(params, doseq=True)
            headers = {}
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(url=url, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if signed and allow_time_resync and "\"code\":-1021" in message:
                self.sync_server_time()
                return self._request(method, path, params=params, signed=signed, allow_time_resync=False)
            raise RuntimeError(f"HTTP {exc.code} for {path}: {message}") from exc

    def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)

    def get_book_ticker(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol}, signed=False)

    def get_open_orders(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v3/openOrders", params=params, signed=True)

    def get_my_trades(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v3/myTrades", params=params, signed=True)

    def place_order(self, params: dict[str, Any], test_only: bool = False) -> dict[str, Any]:
        path = "/api/v3/order/test" if test_only else "/api/v3/order"
        return self._request("POST", path, params=params, signed=True)

    def cancel_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("DELETE", "/api/v3/order", params=params, signed=True)

    def cancel_replace_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v3/order/cancelReplace", params=params, signed=True)


def extract_symbol_rules(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = exchange_info.get("symbols") or []
    if not symbols:
        raise RuntimeError(f"No symbol metadata returned for {symbol}")
    info = symbols[0]
    filters = {item["filterType"]: item for item in info.get("filters", [])}
    return {
        "symbol": info["symbol"],
        "base_asset": info["baseAsset"],
        "quote_asset": info["quoteAsset"],
        "quote_asset_precision": int(info.get("quoteAssetPrecision", 8)),
        "filters": filters,
    }


def build_paper_balance_map(symbol_rules: dict[str, Any], paper_equity_quote: float) -> dict[str, dict[str, Decimal]]:
    base_asset = str(symbol_rules["base_asset"])
    quote_asset = str(symbol_rules["quote_asset"])
    equity_quote = decimal_from(paper_equity_quote)
    return {
        base_asset: {"free": Decimal("0"), "locked": Decimal("0"), "total": Decimal("0")},
        quote_asset: {"free": equity_quote, "locked": Decimal("0"), "total": equity_quote},
    }


def build_balance_map(account: dict[str, Any]) -> dict[str, dict[str, Decimal]]:
    balances: dict[str, dict[str, Decimal]] = {}
    for row in account.get("balances", []):
        free_amt = decimal_from(row.get("free", "0"))
        locked_amt = decimal_from(row.get("locked", "0"))
        balances[str(row["asset"])] = {
            "free": free_amt,
            "locked": locked_amt,
            "total": free_amt + locked_amt,
        }
    return balances


def get_order_style(config: dict[str, Any]) -> str:
    return str(config.get("execution", {}).get("order_style", "market")).lower()


def build_rebalance_snapshot(
    config: dict[str, Any],
    signal_summary: dict[str, Any],
    symbol_rules: dict[str, Any],
    balances: dict[str, dict[str, Decimal]],
    current_price: Decimal,
) -> dict[str, Any]:
    base_asset = str(symbol_rules["base_asset"])
    quote_asset = str(symbol_rules["quote_asset"])
    base_balance = balances.get(base_asset, {})
    quote_balance = balances.get(quote_asset, {})
    current_base_free = base_balance.get("free", Decimal("0"))
    current_base_total = base_balance.get("total", Decimal("0"))
    current_quote_free = quote_balance.get("free", Decimal("0"))
    current_quote_total = quote_balance.get("total", Decimal("0"))
    equity_quote = current_quote_total + (current_base_total * current_price)
    current_allocation = float((current_base_total * current_price / equity_quote)) if equity_quote > 0 else 0.0
    target_allocation = float(signal_summary["target_allocation"])
    target_base_value = equity_quote * decimal_from(target_allocation)
    current_base_value = current_base_total * current_price
    delta_quote = target_base_value - current_base_value

    max_trade_notional = config["risk"].get("max_trade_notional_usdt")
    if max_trade_notional is not None:
        max_trade_notional_dec = decimal_from(max_trade_notional)
        if delta_quote > max_trade_notional_dec:
            delta_quote = max_trade_notional_dec
        elif delta_quote < -max_trade_notional_dec:
            delta_quote = -max_trade_notional_dec

    return {
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "current_price_dec": current_price,
        "current_base_free_qty_dec": current_base_free,
        "current_base_total_qty_dec": current_base_total,
        "current_quote_free_qty_dec": current_quote_free,
        "current_quote_total_qty_dec": current_quote_total,
        "equity_quote_dec": equity_quote,
        "current_allocation": current_allocation,
        "target_allocation": target_allocation,
        "delta_quote_dec": delta_quote,
    }


def serialize_rebalance_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_asset": snapshot["base_asset"],
        "quote_asset": snapshot["quote_asset"],
        "current_price": float(snapshot["current_price_dec"]),
        "current_base_free_qty": float(snapshot["current_base_free_qty_dec"]),
        "current_base_total_qty": float(snapshot["current_base_total_qty_dec"]),
        "current_quote_free_qty": float(snapshot["current_quote_free_qty_dec"]),
        "current_quote_total_qty": float(snapshot["current_quote_total_qty_dec"]),
        "equity_quote": float(snapshot["equity_quote_dec"]),
        "current_allocation": snapshot["current_allocation"],
        "target_allocation": snapshot["target_allocation"],
        "delta_quote": float(snapshot["delta_quote_dec"]),
    }


def plan_spot_market_rebalance(
    config: dict[str, Any],
    signal_summary: dict[str, Any],
    symbol_rules: dict[str, Any],
    balances: dict[str, dict[str, Decimal]],
    current_price: Decimal,
) -> dict[str, Any]:
    snapshot = build_rebalance_snapshot(
        config=config,
        signal_summary=signal_summary,
        symbol_rules=symbol_rules,
        balances=balances,
        current_price=current_price,
    )
    min_rebalance = decimal_from(config["risk"]["min_rebalance_notional_usdt"])
    side = None
    order_params = None
    order_reason = None

    filters = symbol_rules["filters"]
    lot_filter = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    step_size = decimal_from(lot_filter.get("stepSize", "0"))
    min_qty = decimal_from(lot_filter.get("minQty", "0"))
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    min_notional = decimal_from(notional_filter.get("minNotional", "0"))
    min_trade_notional = max(min_rebalance, min_notional)
    delta_quote = snapshot["delta_quote_dec"]
    current_price = snapshot["current_price_dec"]
    current_base_free = snapshot["current_base_free_qty_dec"]
    current_quote_free = snapshot["current_quote_free_qty_dec"]

    if abs(delta_quote) < min_trade_notional:
        order_reason = "rebalance_below_min_notional"
    elif delta_quote > 0:
        side = "BUY"
        quote_qty = min(delta_quote, current_quote_free).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if quote_qty >= min_trade_notional:
            order_params = {
                "symbol": config["symbol"],
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": decimal_to_str(quote_qty),
                "recvWindow": str(config["api"]["recv_window_ms"]),
                "newOrderRespType": "FULL",
            }
        else:
            order_reason = "buy_quote_qty_below_threshold"
    else:
        side = "SELL"
        sell_qty = quantize_down(abs(delta_quote) / current_price, step_size)
        if sell_qty > current_base_free:
            sell_qty = quantize_down(current_base_free, step_size)
        sell_notional = sell_qty * current_price
        if sell_qty >= min_qty and sell_notional >= min_trade_notional:
            order_params = {
                "symbol": config["symbol"],
                "side": "SELL",
                "type": "MARKET",
                "quantity": decimal_to_str(sell_qty),
                "recvWindow": str(config["api"]["recv_window_ms"]),
                "newOrderRespType": "FULL",
            }
        else:
            order_reason = "sell_qty_or_notional_below_threshold"

    plan = serialize_rebalance_snapshot(snapshot)
    plan.update(
        {
            "order_style": "market",
            "side": side,
            "order_reason": order_reason,
            "order_params": order_params,
        }
    )
    return plan


def price_tick_distance(left: Decimal, right: Decimal, tick_size: Decimal) -> int:
    if tick_size <= 0:
        return 0 if left == right else 1
    return int(((left - right).copy_abs() / tick_size).to_integral_value(rounding=ROUND_DOWN))


def remaining_open_quantity(order: dict[str, Any]) -> Decimal:
    orig_qty = decimal_from(order.get("origQty", "0"))
    executed_qty = decimal_from(order.get("executedQty", "0"))
    remaining = orig_qty - executed_qty
    return remaining if remaining > 0 else Decimal("0")


def summarize_open_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "orderId": order.get("orderId"),
        "clientOrderId": order.get("clientOrderId"),
        "origClientOrderId": order.get("origClientOrderId"),
        "side": order.get("side"),
        "type": order.get("type"),
        "status": order.get("status"),
        "price": order.get("price"),
        "origQty": order.get("origQty"),
        "executedQty": order.get("executedQty"),
        "time": order.get("time"),
        "updateTime": order.get("updateTime"),
    }


def summarize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "tradeId": trade.get("id"),
        "orderId": trade.get("orderId"),
        "side": "BUY" if bool(trade.get("isBuyer")) else "SELL",
        "price": trade.get("price"),
        "qty": trade.get("qty"),
        "quoteQty": trade.get("quoteQty"),
        "commission": trade.get("commission"),
        "commissionAsset": trade.get("commissionAsset"),
        "isMaker": bool(trade.get("isMaker")),
        "time": trade.get("time"),
    }


def summarize_fill_batch(trades: list[dict[str, Any]]) -> dict[str, Any]:
    commission_by_asset: dict[str, float] = {}
    buy_base_qty = 0.0
    sell_base_qty = 0.0
    buy_quote_qty = 0.0
    sell_quote_qty = 0.0
    maker_fill_count = 0
    for trade in trades:
        side = "BUY" if bool(trade.get("isBuyer")) else "SELL"
        qty = float(trade.get("qty", 0.0))
        quote_qty = float(trade.get("quoteQty", 0.0))
        commission = float(trade.get("commission", 0.0))
        commission_asset = str(trade.get("commissionAsset", ""))
        if side == "BUY":
            buy_base_qty += qty
            buy_quote_qty += quote_qty
        else:
            sell_base_qty += qty
            sell_quote_qty += quote_qty
        if bool(trade.get("isMaker")):
            maker_fill_count += 1
        if commission_asset:
            commission_by_asset[commission_asset] = commission_by_asset.get(commission_asset, 0.0) + commission
    return {
        "fill_count": len(trades),
        "maker_fill_count": maker_fill_count,
        "buy_base_qty": buy_base_qty,
        "sell_base_qty": sell_base_qty,
        "net_base_qty": buy_base_qty - sell_base_qty,
        "buy_quote_qty": buy_quote_qty,
        "sell_quote_qty": sell_quote_qty,
        "net_quote_qty": sell_quote_qty - buy_quote_qty,
        "commission_by_asset": commission_by_asset,
    }


def collect_order_ids(payload: Any) -> list[int]:
    found: set[int] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "orderId":
                    try:
                        found.add(int(value))
                    except (TypeError, ValueError):
                        pass
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return sorted(found)


def seed_managed_order_ids_from_log(order_log_path: Path, max_ids: int = 200) -> list[int]:
    if not order_log_path.exists():
        return []
    try:
        lines = order_log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    found: list[int] = []
    for line in reversed(lines[-max_ids:]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        for order_id in reversed(collect_order_ids(payload.get("response"))):
            if order_id not in found:
                found.append(order_id)
            if len(found) >= max_ids:
                return list(reversed(found))
    return list(reversed(found))


def build_cycle_status(output: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    signal_summary = output.get("signal_summary") or {}
    execution = output.get("execution") or {}
    primary_interval = str(config.get("strategy", {}).get("primary_interval", ""))
    primary_signal = (output.get("signals") or {}).get(primary_interval, {})
    return {
        "generated_at": output.get("generated_at"),
        "mode": output.get("mode"),
        "strategy_mode": signal_summary.get("strategy_mode", get_strategy_mode(config)),
        "single_interval": signal_summary.get("single_interval"),
        "primary_interval": primary_interval,
        "primary_context_timestamp": primary_signal.get("context_timestamp"),
        "primary_direction": primary_signal.get("direction"),
        "target_allocation": signal_summary.get("target_allocation"),
        "target_exposure": signal_summary.get("target_exposure"),
        "skipped": bool(output.get("skipped")),
        "skip_reason": output.get("skip_reason"),
        "signal_reasons": signal_summary.get("reasons"),
        "execution_kind": execution.get("kind"),
        "execution_action": execution.get("action"),
        "order_sent": execution.get("order_sent"),
        "order_reason": execution.get("order_reason"),
        "managed_open_order_count": execution.get("managed_open_order_count"),
        "new_fill_count": execution.get("new_fill_count"),
        "new_fill_summary": execution.get("new_fill_summary"),
        "account_equity_quote": execution.get("account_equity_quote"),
        "current_allocation": execution.get("current_allocation"),
    }


def format_cycle_status_line(cycle_status: dict[str, Any]) -> str:
    generated_at = str(cycle_status.get("generated_at", ""))
    mode = str(cycle_status.get("mode", ""))
    primary_interval = str(cycle_status.get("primary_interval", ""))
    context_ts = str(cycle_status.get("primary_context_timestamp") or "-")
    direction = str(cycle_status.get("primary_direction") or "-")
    target = cycle_status.get("target_allocation")
    action = str(cycle_status.get("execution_action") or "-")
    fills = int(cycle_status.get("new_fill_count") or 0)
    open_orders = int(cycle_status.get("managed_open_order_count") or 0)
    skipped = bool(cycle_status.get("skipped"))
    skip_reason = str(cycle_status.get("skip_reason") or "")
    return (
        f"[{generated_at}] mode={mode} interval={primary_interval} ctx={context_ts} "
        f"dir={direction} target={target} action={action} fills={fills} open={open_orders} "
        f"skipped={skipped}{' reason=' + skip_reason if skip_reason else ''}"
    )


def select_passive_limit_price(
    side: str,
    best_bid: Decimal,
    best_ask: Decimal,
    tick_size: Decimal,
    inside_spread_ticks: int,
) -> tuple[Decimal | None, str]:
    if best_bid <= 0 or best_ask <= 0:
        return None, "invalid_book_ticker"
    if best_ask <= best_bid:
        return None, "spread_not_positive"

    inside_ticks = max(int(inside_spread_ticks), 1)
    if side == "BUY":
        join_price = quantize_down(best_bid, tick_size)
        inside_price = quantize_down(best_ask - (tick_size * inside_ticks), tick_size)
        if inside_price > join_price and inside_price < best_ask:
            return inside_price, "inside_spread_buy"
        if join_price < best_ask:
            return join_price, "join_best_bid"
        return None, "no_passive_buy_price"

    join_price = quantize_up(best_ask, tick_size)
    inside_price = quantize_up(best_bid + (tick_size * inside_ticks), tick_size)
    if inside_price < join_price and inside_price > best_bid:
        return inside_price, "inside_spread_sell"
    if join_price > best_bid:
        return join_price, "join_best_ask"
    return None, "no_passive_sell_price"


def summarize_book_ticker(book_ticker: dict[str, Any], tick_size: Decimal) -> dict[str, Any]:
    best_bid = decimal_from(book_ticker.get("bidPrice", "0"))
    best_ask = decimal_from(book_ticker.get("askPrice", "0"))
    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / Decimal("2") if best_bid > 0 and best_ask > 0 else Decimal("0")
    return {
        "bid_price": float(best_bid) if best_bid > 0 else None,
        "ask_price": float(best_ask) if best_ask > 0 else None,
        "bid_qty": float(decimal_from(book_ticker.get("bidQty", "0"))),
        "ask_qty": float(decimal_from(book_ticker.get("askQty", "0"))),
        "spread": float(spread) if spread > 0 else 0.0,
        "spread_ticks": price_tick_distance(best_ask, best_bid, tick_size) if spread > 0 else 0,
        "spread_bps": float((spread / mid_price) * Decimal("10000")) if mid_price > 0 and spread > 0 else None,
    }


def is_limit_maker_taker_reject(exc: RuntimeError) -> bool:
    message = str(exc)
    return "\"code\":-2010" in message and "immediately match and take" in message


def build_retry_limit_maker_params(
    client: BinanceSpotClient,
    config: dict[str, Any],
    plan: dict[str, Any],
    order_params: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    side = str(order_params["side"])
    tick_size = decimal_from(plan.get("tick_size", 0))
    fresh_book_ticker = client.get_book_ticker(str(config["symbol"]))
    retry_price, pricing_mode = select_passive_limit_price(
        side=side,
        best_bid=decimal_from(fresh_book_ticker.get("bidPrice", "0")),
        best_ask=decimal_from(fresh_book_ticker.get("askPrice", "0")),
        tick_size=tick_size,
        inside_spread_ticks=int(config.get("execution", {}).get("maker_inside_spread_ticks", 1)),
    )
    if retry_price is None:
        return None, fresh_book_ticker, pricing_mode

    current_price_raw = order_params.get("price")
    current_price = decimal_from(current_price_raw) if current_price_raw is not None else None
    if current_price is not None and retry_price == current_price and tick_size > 0:
        retry_price = retry_price - tick_size if side == "BUY" else retry_price + tick_size
        if retry_price <= 0:
            return None, fresh_book_ticker, "retry_price_non_positive"

    retry_params = copy.deepcopy(order_params)
    retry_params.pop("pegPriceType", None)
    retry_params.pop("pegOffsetType", None)
    retry_params.pop("pegOffsetValue", None)
    retry_params["price"] = decimal_to_str(retry_price)
    return retry_params, fresh_book_ticker, pricing_mode


def plan_spot_maker_limit_rebalance(
    config: dict[str, Any],
    signal_summary: dict[str, Any],
    symbol_rules: dict[str, Any],
    balances: dict[str, dict[str, Decimal]],
    current_price: Decimal,
    book_ticker: dict[str, Any],
) -> dict[str, Any]:
    snapshot = build_rebalance_snapshot(
        config=config,
        signal_summary=signal_summary,
        symbol_rules=symbol_rules,
        balances=balances,
        current_price=current_price,
    )
    filters = symbol_rules["filters"]
    lot_filter = filters.get("LOT_SIZE") or {}
    step_size = decimal_from(lot_filter.get("stepSize", "0"))
    min_qty = decimal_from(lot_filter.get("minQty", "0"))
    price_filter = filters.get("PRICE_FILTER") or {}
    tick_size = decimal_from(price_filter.get("tickSize", "0"))
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    min_notional = decimal_from(notional_filter.get("minNotional", "0"))
    min_rebalance = decimal_from(config["risk"]["min_rebalance_notional_usdt"])
    min_trade_notional = max(min_rebalance, min_notional)

    best_bid = decimal_from(book_ticker.get("bidPrice", "0"))
    best_ask = decimal_from(book_ticker.get("askPrice", "0"))
    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / Decimal("2") if best_bid > 0 and best_ask > 0 else snapshot["current_price_dec"]
    spread_bps = float((spread / mid_price) * Decimal("10000")) if mid_price > 0 and spread > 0 else None
    spread_ticks = price_tick_distance(best_ask, best_bid, tick_size) if spread > 0 else 0

    delta_quote = snapshot["delta_quote_dec"]
    current_base_free = snapshot["current_base_free_qty_dec"]
    current_quote_free = snapshot["current_quote_free_qty_dec"]
    side = "BUY" if delta_quote > 0 else "SELL" if delta_quote < 0 else None
    order_params = None
    order_reason = None
    limit_price: Decimal | None = None
    limit_qty: Decimal | None = None
    pricing_mode = None
    use_primary_peg = bool(config.get("execution", {}).get("use_primary_peg_limit_maker", False))
    peg_offset_levels = max(int(config.get("execution", {}).get("maker_peg_offset_levels", 0)), 0)

    if side is None:
        order_reason = "already_at_target"
    elif abs(delta_quote) < min_trade_notional:
        order_reason = "rebalance_below_min_notional"
    else:
        limit_price, pricing_mode = select_passive_limit_price(
            side=side,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=tick_size,
            inside_spread_ticks=int(config.get("execution", {}).get("maker_inside_spread_ticks", 1)),
        )
        if limit_price is None:
            order_reason = pricing_mode
        elif side == "BUY":
            buy_buffer_pct = max(float(config.get("execution", {}).get("maker_buy_balance_buffer_pct", 0.25)), 0.0) / 100.0
            spendable_quote = min(delta_quote.copy_abs(), current_quote_free * decimal_from(max(0.0, 1.0 - buy_buffer_pct)))
            limit_qty = quantize_down(spendable_quote / limit_price, step_size)
            limit_notional = limit_qty * limit_price
            if limit_qty < min_qty:
                order_reason = "buy_qty_below_lot_size"
            elif limit_notional < min_trade_notional:
                order_reason = "buy_notional_below_threshold"
            else:
                order_params = {
                    "symbol": config["symbol"],
                    "side": side,
                    "type": "LIMIT_MAKER",
                    "quantity": decimal_to_str(limit_qty),
                    "recvWindow": str(config["api"]["recv_window_ms"]),
                    "newOrderRespType": "RESULT",
                }
                if use_primary_peg:
                    order_params["pegPriceType"] = "PRIMARY_PEG"
                    if peg_offset_levels > 0:
                        order_params["pegOffsetType"] = "PRICE_LEVEL"
                        order_params["pegOffsetValue"] = str(peg_offset_levels)
                else:
                    order_params["price"] = decimal_to_str(limit_price)
        else:
            desired_qty = quantize_down(delta_quote.copy_abs() / limit_price, step_size)
            available_qty = quantize_down(current_base_free, step_size)
            limit_qty = min(desired_qty, available_qty)
            limit_notional = limit_qty * limit_price
            if current_base_free <= 0:
                order_reason = "no_free_base_to_sell"
            elif limit_qty < min_qty:
                order_reason = "sell_qty_below_lot_size"
            elif limit_notional < min_trade_notional:
                order_reason = "sell_notional_below_threshold"
            else:
                order_params = {
                    "symbol": config["symbol"],
                    "side": side,
                    "type": "LIMIT_MAKER",
                    "quantity": decimal_to_str(limit_qty),
                    "recvWindow": str(config["api"]["recv_window_ms"]),
                    "newOrderRespType": "RESULT",
                }
                if use_primary_peg:
                    order_params["pegPriceType"] = "PRIMARY_PEG"
                    if peg_offset_levels > 0:
                        order_params["pegOffsetType"] = "PRICE_LEVEL"
                        order_params["pegOffsetValue"] = str(peg_offset_levels)
                else:
                    order_params["price"] = decimal_to_str(limit_price)

    plan = serialize_rebalance_snapshot(snapshot)
    plan.update(
        {
            "order_style": "maker_limit",
            "side": side,
            "order_reason": order_reason,
            "order_params": order_params,
            "limit_price": float(limit_price) if limit_price is not None else None,
            "limit_quantity": float(limit_qty) if limit_qty is not None else None,
            "pricing_mode": pricing_mode,
            "book_ticker": summarize_book_ticker(book_ticker, tick_size),
            "tick_size": float(tick_size),
            "step_size": float(step_size),
            "min_trade_notional_quote": float(min_trade_notional),
            "use_primary_peg_limit_maker": use_primary_peg,
        }
    )
    return plan


def build_managed_client_order_id(prefix: str, side: str) -> str:
    side_tag = "B" if side == "BUY" else "S"
    timestamp_tag = utc_now().strftime("%y%m%d%H%M%S")
    millis_tag = f"{utc_now_ms() % 100000:05d}"
    return f"{prefix}-{side_tag}-{timestamp_tag}-{millis_tag}"[:36]


def filter_managed_orders(open_orders: list[dict[str, Any]], client_order_prefix: str) -> list[dict[str, Any]]:
    return [order for order in open_orders if str(order.get("clientOrderId", "")).startswith(client_order_prefix)]


def choose_primary_managed_order(
    orders: list[dict[str, Any]],
    desired_price: Decimal,
    desired_qty: Decimal,
    tick_size: Decimal,
) -> dict[str, Any] | None:
    if not orders:
        return None

    def sort_key(order: dict[str, Any]) -> tuple[Any, ...]:
        order_price = decimal_from(order.get("price", "0"))
        remaining_qty = remaining_open_quantity(order)
        updated_at = int(order.get("updateTime") or order.get("time") or 0)
        return (
            price_tick_distance(order_price, desired_price, tick_size),
            abs(remaining_qty - desired_qty),
            -updated_at,
        )

    return min(orders, key=sort_key)


def should_replace_managed_order(
    order: dict[str, Any],
    desired_price: Decimal,
    desired_qty: Decimal,
    tick_size: Decimal,
    step_size: Decimal,
    reprice_after_ticks: int,
) -> tuple[bool, str]:
    if str(order.get("type", "")) != "LIMIT_MAKER":
        return True, "existing_order_not_limit_maker"
    existing_price = decimal_from(order.get("price", "0"))
    existing_remaining_qty = remaining_open_quantity(order)
    if price_tick_distance(existing_price, desired_price, tick_size) >= max(reprice_after_ticks, 1):
        return True, "limit_price_stale"
    qty_threshold = step_size if step_size > 0 else Decimal("0")
    if existing_remaining_qty != desired_qty and (qty_threshold <= 0 or abs(existing_remaining_qty - desired_qty) >= qty_threshold):
        return True, "limit_quantity_changed"
    return False, "existing_order_matches_target"


def execute_maker_limit_plan(
    client: BinanceSpotClient,
    config: dict[str, Any],
    plan: dict[str, Any],
    test_order: bool,
) -> dict[str, Any]:
    recv_window = str(config["api"]["recv_window_ms"])
    symbol = str(config["symbol"])
    prefix = str(config.get("execution", {}).get("maker_client_order_prefix", "kronosmkr"))
    side = plan.get("side")
    order_params = copy.deepcopy(plan.get("order_params"))
    result: dict[str, Any] = {
        "order_sent": False,
        "test_order": bool(test_order),
        "action": "no_action",
        "response": None,
        "cancel_responses": [],
        "managed_open_orders": [],
    }

    if not order_params and test_order:
        result["action"] = "no_order_to_test"
        return result
    if order_params and test_order:
        params = copy.deepcopy(order_params)
        params["newClientOrderId"] = build_managed_client_order_id(prefix, str(side))
        result["response"] = client.place_order(params, test_only=True)
        result["order_sent"] = True
        result["action"] = "test_new_limit_maker"
        return result

    open_orders = client.get_open_orders({"symbol": symbol, "recvWindow": recv_window})
    managed_orders = filter_managed_orders(open_orders, prefix)
    result["managed_open_orders"] = [summarize_open_order(order) for order in managed_orders]

    desired_price = None
    if order_params:
        desired_price_raw = order_params.get("price", plan.get("limit_price"))
        if desired_price_raw is not None:
            desired_price = decimal_from(desired_price_raw)
    desired_qty = decimal_from(order_params["quantity"]) if order_params else None
    tick_size = decimal_from(plan.get("tick_size", 0))
    step_size = decimal_from(plan.get("step_size", 0))
    reprice_after_ticks = int(config.get("execution", {}).get("maker_reprice_after_ticks", 1))
    cancel_on_idle = bool(config.get("execution", {}).get("maker_cancel_on_idle", True))
    cancel_on_signal_flip = bool(config.get("execution", {}).get("maker_cancel_on_signal_flip", True))
    preserve_existing_same_side_reasons = {
        "buy_qty_below_lot_size",
        "buy_notional_below_threshold",
        "sell_qty_below_lot_size",
        "sell_notional_below_threshold",
        "no_free_base_to_sell",
    }
    preserve_existing_same_side = (
        not order_params
        and bool(side)
        and str(plan.get("order_reason") or "") in preserve_existing_same_side_reasons
    )

    def cancel_order(order: dict[str, Any], reason: str) -> None:
        cancel_params = {
            "symbol": symbol,
            "orderId": str(order["orderId"]),
            "recvWindow": recv_window,
        }
        response = client.cancel_order(cancel_params)
        result["cancel_responses"].append({"reason": reason, "response": response})

    same_side_orders: list[dict[str, Any]] = []
    for managed_order in managed_orders:
        order_side = str(managed_order.get("side", ""))
        if not order_params:
            if preserve_existing_same_side and order_side == side:
                same_side_orders.append(managed_order)
                continue
            if cancel_on_idle:
                cancel_order(managed_order, "target_has_no_live_order")
            continue
        if order_side != side:
            if cancel_on_signal_flip:
                cancel_order(managed_order, "signal_side_changed")
            continue
        same_side_orders.append(managed_order)

    if not order_params:
        if same_side_orders:
            result["action"] = "keep_existing_limit_maker"
            result["kept_order"] = summarize_open_order(same_side_orders[0])
            return result
        result["action"] = "cancel_idle_orders" if result["cancel_responses"] else "idle_without_order"
        return result

    primary_order = None
    if desired_price is not None and desired_qty is not None:
        primary_order = choose_primary_managed_order(same_side_orders, desired_price, desired_qty, tick_size)
    for order in same_side_orders:
        if primary_order is None or order.get("orderId") != primary_order.get("orderId"):
            cancel_order(order, "duplicate_managed_order")

    if primary_order is not None and desired_price is not None and desired_qty is not None:
        replace_needed, replace_reason = should_replace_managed_order(
            order=primary_order,
            desired_price=desired_price,
            desired_qty=desired_qty,
            tick_size=tick_size,
            step_size=step_size,
            reprice_after_ticks=reprice_after_ticks,
        )
        if not replace_needed:
            result["action"] = "keep_existing_limit_maker"
            result["kept_order"] = summarize_open_order(primary_order)
            return result

        replace_params = copy.deepcopy(order_params)
        replace_params.update(
            {
                "cancelReplaceMode": "STOP_ON_FAILURE",
                "cancelOrderId": str(primary_order["orderId"]),
                "newClientOrderId": build_managed_client_order_id(prefix, str(side)),
            }
        )
        result["response"] = client.cancel_replace_order(replace_params)
        result["order_sent"] = True
        result["action"] = "cancel_replace_limit_maker"
        result["replace_reason"] = replace_reason
        return result

    new_params = copy.deepcopy(order_params)
    new_params["newClientOrderId"] = build_managed_client_order_id(prefix, str(side))
    try:
        result["response"] = client.place_order(new_params, test_only=False)
        result["order_sent"] = True
        result["action"] = "new_limit_maker"
    except RuntimeError as exc:
        if not is_limit_maker_taker_reject(exc):
            raise
        retry_params, retry_book_ticker, retry_pricing_mode = build_retry_limit_maker_params(
            client=client,
            config=config,
            plan=plan,
            order_params=new_params,
        )
        result["retry_reason"] = "limit_maker_would_take"
        result["retry_book_ticker"] = summarize_book_ticker(retry_book_ticker, decimal_from(plan.get("tick_size", 0)))
        result["retry_pricing_mode"] = retry_pricing_mode
        if retry_params is None:
            raise
        result["response"] = client.place_order(retry_params, test_only=False)
        result["order_sent"] = True
        result["action"] = "retry_new_limit_maker"
        result["order_params"] = retry_params
    return result


def enrich_execution_with_live_status(
    client: BinanceSpotClient,
    config: dict[str, Any],
    state: dict[str, Any],
    execution: dict[str, Any],
    order_log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if execution.get("kind") != "spot":
        return execution, state

    recv_window = str(config["api"]["recv_window_ms"])
    symbol = str(config["symbol"])
    prefix = str(config.get("execution", {}).get("maker_client_order_prefix", "kronosmkr"))
    tracked_order_ids = set(int(value) for value in state.get("managed_order_ids", []) if str(value).strip())
    if not tracked_order_ids:
        tracked_order_ids.update(seed_managed_order_ids_from_log(order_log_path))
    tracked_order_ids.update(collect_order_ids(execution.get("response")))

    if execution.get("order_style") == "maker_limit":
        open_orders = client.get_open_orders({"symbol": symbol, "recvWindow": recv_window})
        managed_open_orders = filter_managed_orders(open_orders, prefix)
        execution["managed_open_orders"] = [summarize_open_order(order) for order in managed_open_orders]
        execution["managed_open_order_count"] = len(managed_open_orders)
        tracked_order_ids.update(int(order["orderId"]) for order in managed_open_orders if order.get("orderId") is not None)
    else:
        execution.setdefault("managed_open_orders", [])
        execution.setdefault("managed_open_order_count", 0)

    last_fill_time_ms = int(state.get("last_seen_managed_fill_time_ms", 0))
    last_fill_trade_id = int(state.get("last_seen_managed_trade_id", 0))
    new_fills: list[dict[str, Any]] = []
    if tracked_order_ids:
        recent_trades = client.get_my_trades({"symbol": symbol, "limit": 100, "recvWindow": recv_window})
        for trade in recent_trades:
            order_id = int(trade.get("orderId", 0))
            trade_time = int(trade.get("time", 0))
            trade_id = int(trade.get("id", 0))
            if order_id not in tracked_order_ids:
                continue
            if trade_time < last_fill_time_ms:
                continue
            if trade_time == last_fill_time_ms and trade_id <= last_fill_trade_id:
                continue
            new_fills.append(trade)
        new_fills.sort(key=lambda trade: (int(trade.get("time", 0)), int(trade.get("id", 0))))
        if new_fills:
            last_trade = new_fills[-1]
            state["last_seen_managed_fill_time_ms"] = int(last_trade.get("time", 0))
            state["last_seen_managed_trade_id"] = int(last_trade.get("id", 0))

    state["managed_order_ids"] = sorted(tracked_order_ids)[-200:]
    execution["tracked_managed_order_count"] = len(state["managed_order_ids"])
    execution["new_fills"] = [summarize_trade(trade) for trade in new_fills]
    execution["new_fill_count"] = len(new_fills)
    execution["new_fill_summary"] = summarize_fill_batch(new_fills)
    return execution, state


def plan_spot_rebalance(
    config: dict[str, Any],
    signal_summary: dict[str, Any],
    symbol_rules: dict[str, Any],
    balances: dict[str, dict[str, Decimal]],
    current_price: Decimal,
    book_ticker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    order_style = get_order_style(config)
    if order_style == "maker_limit":
        if book_ticker is None:
            raise RuntimeError("Maker limit execution requires a live book ticker snapshot")
        return plan_spot_maker_limit_rebalance(
            config=config,
            signal_summary=signal_summary,
            symbol_rules=symbol_rules,
            balances=balances,
            current_price=current_price,
            book_ticker=book_ticker,
        )
    return plan_spot_market_rebalance(
        config=config,
        signal_summary=signal_summary,
        symbol_rules=symbol_rules,
        balances=balances,
        current_price=current_price,
    )


def update_risk_state(state: dict[str, Any], config: dict[str, Any], equity_quote: float) -> dict[str, Any]:
    today = utc_now().date().isoformat()
    if state.get("risk_date") != today:
        state["risk_date"] = today
        state["daily_start_equity"] = equity_quote
        state["intraday_high_equity"] = equity_quote
        state["risk_halt"] = False
        state["risk_halt_reason"] = None
    else:
        state["intraday_high_equity"] = max(float(state.get("intraday_high_equity", equity_quote)), equity_quote)

    daily_start = float(state.get("daily_start_equity", equity_quote))
    intraday_high = float(state.get("intraday_high_equity", equity_quote))
    daily_limit = float(config["risk"]["max_daily_drawdown_pct"]) / 100.0
    intraday_limit = float(config["risk"]["max_intraday_drawdown_pct"]) / 100.0

    risk_halt = False
    risk_halt_reason = None
    if daily_start > 0 and equity_quote < daily_start * (1.0 - daily_limit):
        risk_halt = True
        risk_halt_reason = "Daily drawdown limit breached"
    if intraday_high > 0 and equity_quote < intraday_high * (1.0 - intraday_limit):
        risk_halt = True
        risk_halt_reason = "Intraday drawdown limit breached"

    state["risk_halt"] = risk_halt
    state["risk_halt_reason"] = risk_halt_reason
    return state


def current_primary_context_timestamp(signals: dict[str, ModelSignal], primary_interval: str) -> str:
    return signals[primary_interval].context_timestamp


def maybe_skip_same_primary_bar(state: dict[str, Any], config: dict[str, Any], signals: dict[str, ModelSignal], force: bool) -> tuple[bool, str | None]:
    if force:
        return False, None
    primary_interval = str(config["strategy"]["primary_interval"])
    current_primary = current_primary_context_timestamp(signals, primary_interval)
    if state.get("last_primary_context_timestamp") == current_primary:
        return True, "Primary interval has not closed a new bar yet"
    return False, None


def resolve_base_url(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized == "testnet":
        return TESTNET_SPOT_BASE_URL
    if normalized == "demo":
        return DEMO_SPOT_BASE_URL
    return LIVE_SPOT_BASE_URL


def resolve_stream_base_url(mode: str) -> str | None:
    normalized = str(mode).lower()
    if normalized == "testnet":
        return TESTNET_SPOT_STREAM_BASE_URL
    if normalized in {"dry_run", "live"}:
        return LIVE_SPOT_STREAM_BASE_URL
    return None


def resolve_ws_api_url(mode: str) -> str | None:
    normalized = str(mode).lower()
    if normalized == "testnet":
        return TESTNET_SPOT_WS_API_URL
    if normalized == "live":
        return LIVE_SPOT_WS_API_URL
    return None


def is_single_interval_1m_strategy(config: dict[str, Any]) -> bool:
    strategy = config.get("strategy", {})
    return (
        get_strategy_mode(config) == "single_interval"
        and str(strategy.get("single_interval", "")).strip() == "1m"
        and str(strategy.get("primary_interval", "")).strip() == "1m"
    )


def is_fast_1m_enabled(config: dict[str, Any]) -> bool:
    return is_single_interval_1m_strategy(config) and bool(config.get("execution", {}).get("fast_1m_enabled", False))


def build_ws_api_signature_payload(params: dict[str, Any]) -> str:
    return "&".join(f"{key}={params[key]}" for key in sorted(params.keys()) if key != "signature")


def sign_ws_api_params(api_secret: str, params: dict[str, Any]) -> str:
    payload = build_ws_api_signature_payload(params)
    return hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def resolve_runtime_device(device: str | None) -> str:
    normalized = str(device or "").strip().lower()
    if normalized in {"", "auto", "cuda_if_available"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def summarize_signals(signals: dict[str, ModelSignal]) -> dict[str, Any]:
    return {
        key: {
            "model_id": value.model_id,
            "model_name": value.model_name,
            "family": value.family,
            "role": value.role,
            "metric_name": value.metric_name,
            "interval": value.interval,
            "direction": value.direction,
            "signal_value": value.signal_value,
            "confidence_pct": value.confidence_pct,
            "move_pct": value.move_pct,
            "prob_up": value.prob_up,
            "reference_price": value.reference_price,
            "predicted_price": value.predicted_price,
            "context_timestamp": value.context_timestamp,
            "target_timestamp": value.target_timestamp,
            "recent_vol_pct": value.recent_vol_pct,
            "market_path": value.market_path,
        }
        for key, value in signals.items()
    }


def build_dry_run_portfolio_preview(config: dict[str, Any], signal_summary: dict[str, Any], current_price: float) -> dict[str, Any]:
    equity = float(config["risk"]["default_paper_equity_usdt"])
    current_allocation = 0.0
    target_allocation = float(signal_summary["target_allocation"])
    delta_quote = equity * (target_allocation - current_allocation)
    return {
        "equity_quote": equity,
        "current_allocation": current_allocation,
        "target_exposure": float(signal_summary.get("target_exposure", target_allocation)),
        "target_allocation": target_allocation,
        "spot_execution_clips_short": bool(signal_summary.get("target_exposure", 0.0) < 0.0),
        "delta_quote": delta_quote,
        "current_price": current_price,
    }


def resolve_current_price(config: dict[str, Any]) -> Decimal:
    preferred_keys: list[str] = []
    primary_interval = str(config.get("strategy", {}).get("primary_interval", "")).strip()
    if primary_interval:
        preferred_keys.append(primary_interval)
    preferred_keys.extend(key for key in get_active_model_keys(config) if key not in preferred_keys)
    preferred_keys.extend(key for key in ("1m", "15m", "1h", "4h", "1d") if key not in preferred_keys)
    for interval_key in preferred_keys:
        if interval_key not in config["models"]:
            continue
        market_path = build_market_path(config, config["models"][interval_key])
        if market_path.exists():
            market_df = load_market_dataframe(market_path)
            if not market_df.empty:
                return decimal_from(market_df["close"].iloc[-1])
    raise RuntimeError("Could not resolve a current market price from cached interval files")


class FastMinuteMarketDataFeed:
    def __init__(self, stream_base_url: str, base_url: str, symbol: str, interval_ms: int = 60_000):
        self.stream_base_url = stream_base_url.rstrip("/")
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol.upper()
        self.symbol_lower = self.symbol.lower()
        self.interval_ms = int(interval_ms)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._book_ticker: dict[str, Any] | None = None
        self._current_bar: dict[str, Any] | None = None
        self._last_closed_bar: dict[str, Any] | None = None
        self._last_closed_open_ms: int | None = None
        self._last_trade_price: Decimal | None = None
        self._last_trade_time_ms = 0
        self._server_time_offset_ms = 0
        self._last_time_sync_local_ms = 0
        self._last_stream_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=f"fast-1m-feed-{self.symbol_lower}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def sync_server_time(self) -> int:
        server_time_ms = fetch_exchange_server_time_ms(self.base_url)
        local_time_ms = utc_now_ms()
        with self._lock:
            self._server_time_offset_ms = server_time_ms - local_time_ms
            self._last_time_sync_local_ms = local_time_ms
            offset_ms = self._server_time_offset_ms
        update_exchange_time_offset(offset_ms, synced_local_ms=local_time_ms)
        return offset_ms

    def ensure_fresh_server_time(self, max_age_ms: int = 300_000) -> int:
        with self._lock:
            last_sync_local_ms = self._last_time_sync_local_ms
            offset_ms = self._server_time_offset_ms
        if last_sync_local_ms == 0 or abs(utc_now_ms() - last_sync_local_ms) > max(int(max_age_ms), 1):
            return self.sync_server_time()
        return offset_ms

    def server_now_ms(self) -> int:
        with self._lock:
            offset_ms = self._server_time_offset_ms
        return utc_now_ms() + offset_ms

    def wait_until_ready(self, timeout_seconds: float = 15.0) -> bool:
        return self._ready_event.wait(timeout=max(float(timeout_seconds), 0.1))

    def last_stream_error(self) -> str | None:
        with self._lock:
            return self._last_stream_error

    def get_book_ticker(self, symbol: str | None = None) -> dict[str, Any] | None:
        if symbol and str(symbol).upper() != self.symbol:
            return None
        with self._lock:
            return copy.deepcopy(self._book_ticker)

    def get_last_trade_price(self) -> Decimal | None:
        with self._lock:
            return Decimal(str(self._last_trade_price)) if self._last_trade_price is not None else None

    def finalize_closed_bar(self, close_boundary_ms: int) -> dict[str, Any] | None:
        target_open_ms = int(close_boundary_ms) - self.interval_ms
        with self._lock:
            if self._last_closed_open_ms == target_open_ms and self._last_closed_bar is not None:
                return copy.deepcopy(self._last_closed_bar)
            if self._current_bar is None:
                return None

            while self._current_bar is not None and int(self._current_bar["open_time_ms"]) < int(close_boundary_ms):
                finalized_bar = copy.deepcopy(self._current_bar)
                finalized_bar["close_time_ms"] = int(finalized_bar["open_time_ms"]) + self.interval_ms - 1
                self._last_closed_bar = finalized_bar
                self._last_closed_open_ms = int(finalized_bar["open_time_ms"])
                carry_price = decimal_from(finalized_bar["close"])
                next_open_ms = int(finalized_bar["open_time_ms"]) + self.interval_ms
                self._current_bar = self._build_seed_bar(next_open_ms, carry_price)
                if self._last_closed_open_ms == target_open_ms:
                    return copy.deepcopy(self._last_closed_bar)

            if self._last_closed_open_ms == target_open_ms and self._last_closed_bar is not None:
                return copy.deepcopy(self._last_closed_bar)
            return None

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect as websocket_connect
        except Exception as exc:  # pragma: no cover - environment guard
            with self._lock:
                self._last_stream_error = f"websockets import failed: {exc}"
            return

        stream_url = (
            f"{self.stream_base_url}/stream?streams="
            f"{self.symbol_lower}@trade/{self.symbol_lower}@bookTicker"
        )
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            try:
                with websocket_connect(
                    stream_url,
                    ping_interval=20,
                    ping_timeout=60,
                    open_timeout=15,
                    close_timeout=10,
                    max_size=None,
                    proxy=None,
                ) as websocket:
                    backoff_seconds = 1.0
                    for message in websocket:
                        if self._stop_event.is_set():
                            return
                        payload = json.loads(message)
                        event = payload.get("data", payload)
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("e")
                        if event_type == "trade":
                            self._handle_trade_event(event)
                        elif event_type == "bookTicker" or {"u", "s", "b", "B", "a", "A"}.issubset(event.keys()):
                            self._handle_book_ticker_event(event)
            except Exception as exc:  # pragma: no cover - operational guard
                with self._lock:
                    self._last_stream_error = str(exc)
                if self._stop_event.is_set():
                    return
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 30.0)

    def _handle_book_ticker_event(self, payload: dict[str, Any]) -> None:
        snapshot = {
            "symbol": payload.get("s", self.symbol),
            "bidPrice": payload.get("b", "0"),
            "bidQty": payload.get("B", "0"),
            "askPrice": payload.get("a", "0"),
            "askQty": payload.get("A", "0"),
            "updateId": payload.get("u"),
            "source": "stream",
            "receivedAt": utc_now_ms(),
        }
        with self._lock:
            self._book_ticker = snapshot
            if self._last_trade_time_ms > 0:
                self._ready_event.set()

    def _handle_trade_event(self, payload: dict[str, Any]) -> None:
        trade_time_ms = int(payload.get("T") or payload.get("E") or 0)
        price = decimal_from(payload.get("p", "0"))
        quantity = decimal_from(payload.get("q", "0"))
        if trade_time_ms <= 0 or price <= 0 or quantity <= 0:
            return
        trade_open_ms = trade_time_ms - (trade_time_ms % self.interval_ms)

        with self._lock:
            self._last_trade_price = price
            self._last_trade_time_ms = trade_time_ms
            if self._current_bar is None:
                self._current_bar = self._build_seed_bar(trade_open_ms, price)
            if trade_open_ms < int(self._current_bar["open_time_ms"]):
                return
            while trade_open_ms > int(self._current_bar["open_time_ms"]):
                finalized_bar = copy.deepcopy(self._current_bar)
                finalized_bar["close_time_ms"] = int(finalized_bar["open_time_ms"]) + self.interval_ms - 1
                self._last_closed_bar = finalized_bar
                self._last_closed_open_ms = int(finalized_bar["open_time_ms"])
                carry_price = decimal_from(finalized_bar["close"])
                self._current_bar = self._build_seed_bar(int(finalized_bar["open_time_ms"]) + self.interval_ms, carry_price)
            self._current_bar["high"] = max(decimal_from(self._current_bar["high"]), price)
            self._current_bar["low"] = min(decimal_from(self._current_bar["low"]), price)
            self._current_bar["close"] = price
            self._current_bar["volume"] = decimal_from(self._current_bar["volume"]) + quantity
            self._current_bar["amount"] = decimal_from(self._current_bar["amount"]) + (price * quantity)
            self._current_bar["trade_count"] = int(self._current_bar.get("trade_count", 0)) + 1
            self._current_bar["last_trade_time_ms"] = trade_time_ms
            if self._book_ticker is not None:
                self._ready_event.set()

    def _build_seed_bar(self, open_time_ms: int, seed_price: Decimal) -> dict[str, Any]:
        seed = decimal_from(seed_price)
        return {
            "open_time_ms": int(open_time_ms),
            "close_time_ms": int(open_time_ms) + self.interval_ms - 1,
            "open": seed,
            "high": seed,
            "low": seed,
            "close": seed,
            "volume": Decimal("0"),
            "amount": Decimal("0"),
            "trade_count": 0,
            "last_trade_time_ms": 0,
        }


class BinanceSpotWsApiClient:
    def __init__(self, ws_api_url: str, base_url: str, api_key: str, api_secret: str, timeout_seconds: int = 10):
        self.ws_api_url = ws_api_url.rstrip("/")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.timeout_seconds = max(int(timeout_seconds), 1)
        self.time_offset_ms = 0
        self.last_time_sync_ms = 0
        self._lock = threading.Lock()
        self._connection: Any | None = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def sync_server_time(self) -> int:
        server_time_ms = fetch_exchange_server_time_ms(self.base_url)
        local_time_ms = utc_now_ms()
        self.time_offset_ms = server_time_ms - local_time_ms
        self.last_time_sync_ms = local_time_ms
        update_exchange_time_offset(self.time_offset_ms, synced_local_ms=local_time_ms)
        return self.time_offset_ms

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._signed_request("order.place", params)

    def cancel_replace_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._signed_request("order.cancelReplace", params)

    def _signed_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("WebSocket API order entry requires API key and secret")
        if self.last_time_sync_ms == 0 or abs(utc_now_ms() - self.last_time_sync_ms) > 300_000:
            self.sync_server_time()
        signed_params = {key: value for key, value in params.items() if value is not None}
        signed_params["apiKey"] = self.api_key
        signed_params["timestamp"] = utc_now_ms() + self.time_offset_ms
        signed_params["signature"] = sign_ws_api_params(self.api_secret, signed_params)
        request_id = str(uuid.uuid4())
        request_payload = {"id": request_id, "method": method, "params": signed_params}

        last_error: Exception | None = None
        for _ in range(2):
            with self._lock:
                try:
                    self._ensure_connection_locked()
                    assert self._connection is not None
                    self._connection.send(json.dumps(request_payload))
                    while True:
                        raw_message = self._connection.recv(timeout=float(self.timeout_seconds))
                        payload = json.loads(raw_message)
                        if payload.get("id") != request_id:
                            continue
                        status_code = int(payload.get("status", 0))
                        if status_code != 200:
                            raise RuntimeError(json.dumps(payload, sort_keys=True))
                        return payload
                except Exception as exc:
                    last_error = exc
                    self._close_locked()
        raise RuntimeError(f"WebSocket API request failed for {method}: {last_error}")

    def _ensure_connection_locked(self) -> None:
        if self._connection is not None:
            return
        from websockets.sync.client import connect as websocket_connect

        self._connection = websocket_connect(
            self.ws_api_url,
            ping_interval=20,
            ping_timeout=60,
            open_timeout=15,
            close_timeout=10,
            max_size=None,
            proxy=None,
        )

    def _close_locked(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        except Exception:
            pass
        self._connection = None


class HybridBinanceSpotClient:
    def __init__(
        self,
        rest_client: BinanceSpotClient,
        ws_order_client: BinanceSpotWsApiClient | None = None,
        book_ticker_provider: Any | None = None,
    ):
        self.rest_client = rest_client
        self.ws_order_client = ws_order_client
        self.book_ticker_provider = book_ticker_provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rest_client, name)

    def get_book_ticker(self, symbol: str) -> dict[str, Any]:
        if self.book_ticker_provider is not None:
            book_ticker = self.book_ticker_provider(symbol)
            if book_ticker is not None:
                return book_ticker
        return self.rest_client.get_book_ticker(symbol)

    def place_order(self, params: dict[str, Any], test_only: bool = False) -> dict[str, Any]:
        if test_only or self.ws_order_client is None:
            return self.rest_client.place_order(params, test_only=test_only)
        try:
            return self.ws_order_client.place_order(params)
        except Exception:
            return self.rest_client.place_order(params, test_only=False)

    def cancel_replace_order(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.ws_order_client is None:
            return self.rest_client.cancel_replace_order(params)
        try:
            return self.ws_order_client.cancel_replace_order(params)
        except Exception:
            return self.rest_client.cancel_replace_order(params)

    def close(self) -> None:
        if self.ws_order_client is not None:
            self.ws_order_client.close()


def build_live_spot_client(
    config: dict[str, Any],
    mode: str,
    *,
    enable_ws_orders: bool = False,
    book_ticker_provider: Any | None = None,
) -> HybridBinanceSpotClient | BinanceSpotClient:
    base_url = resolve_base_url(mode)
    api_key = os.getenv(str(config["api"]["key_env"]), "")
    api_secret = os.getenv(str(config["api"]["secret_env"]), "")
    rest_client = BinanceSpotClient(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
        timeout_seconds=int(config["api"]["timeout_seconds"]),
    )
    ws_order_client = None
    if enable_ws_orders and api_key and api_secret:
        ws_api_url = resolve_ws_api_url(mode)
        if ws_api_url is None:
            raise RuntimeError(f"WebSocket order entry is not available for mode={mode}")
        ws_order_client = BinanceSpotWsApiClient(
            ws_api_url=ws_api_url,
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
            timeout_seconds=min(int(config["api"]["timeout_seconds"]), 10),
        )
    if ws_order_client is not None or book_ticker_provider is not None:
        return HybridBinanceSpotClient(
            rest_client=rest_client,
            ws_order_client=ws_order_client,
            book_ticker_provider=book_ticker_provider,
        )
    return rest_client


def configure_fast_1m_inference(config: dict[str, Any]) -> str:
    spec = config["models"]["1m"]
    requested_device = str(config.get("execution", {}).get("fast_1m_inference_device", spec.get("device", "auto"))).strip()
    resolved_device = resolve_runtime_device(requested_device)
    spec["device"] = resolved_device
    return resolved_device


def build_fast_1m_seed_cache(config: dict[str, Any], base_url: str) -> tuple[Path, int]:
    spec = config["models"]["1m"]
    rows_needed = int(spec.get("lookback", 256)) + int(spec.get("pred_len", 1)) + 8
    market_path = build_market_path(config, spec)
    fetch_latest_closed_klines(
        base_url=base_url,
        symbol=str(config["symbol"]),
        interval=str(spec["binance_interval"]),
        rows_needed=rows_needed,
        output_path=market_path,
        fallback_paths=list(spec.get("fallback_market_paths", [])),
    )
    keep_rows = max(rows_needed + 64, 512)
    return market_path, keep_rows


def prewarm_fast_1m_inference(config: dict[str, Any], market_path: Path) -> dict[str, Any]:
    spec = config["models"]["1m"]
    started_at = time.perf_counter()

    if spec.get("stabilize_output", True) and spec.get("stability_context_path"):
        load_stability_profile(str(spec["stability_context_path"]))

    signal = generate_consecutive_signal(config, spec, market_path)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "device": str(spec.get("device", "")),
        "seconds": round(elapsed_seconds, 3),
        "direction": signal.direction,
        "context_timestamp": signal.context_timestamp,
    }


def upsert_fast_closed_bar(path: Path, closed_bar: dict[str, Any], keep_rows: int) -> None:
    timestamp = pd.to_datetime(int(closed_bar["open_time_ms"]), unit="ms")
    row = {
        "timestamp": timestamp,
        "open": float(decimal_from(closed_bar["open"])),
        "close": float(decimal_from(closed_bar["close"])),
        "high": float(decimal_from(closed_bar["high"])),
        "low": float(decimal_from(closed_bar["low"])),
        "volume": float(decimal_from(closed_bar["volume"])),
        "amount": float(decimal_from(closed_bar["amount"])),
    }
    if path.exists():
        market_df = load_market_dataframe(path)
        market_df = market_df[["timestamp", "open", "close", "high", "low", "volume", "amount"]].copy()
    else:
        market_df = pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume", "amount"])
    market_df = market_df[market_df["timestamp"] < timestamp].copy()
    market_df = pd.concat([market_df, pd.DataFrame([row])], ignore_index=True)
    if keep_rows > 0 and len(market_df) > keep_rows:
        market_df = market_df.tail(keep_rows).reset_index(drop=True)
    write_market_dataframe(path, market_df)


def sleep_until_server_time(feed: FastMinuteMarketDataFeed, target_server_ms: int, max_sleep_ms: int = 250) -> None:
    while True:
        remaining_ms = int(target_server_ms) - int(feed.server_now_ms())
        if remaining_ms <= 0:
            return
        time.sleep(min(remaining_ms, max(int(max_sleep_ms), 1)) / 1000.0)


def sleep_until_exchange_time(target_exchange_ms: int, max_sleep_ms: int = 250) -> None:
    while True:
        remaining_ms = int(target_exchange_ms) - int(exchange_utc_now_ms())
        if remaining_ms <= 0:
            return
        time.sleep(min(remaining_ms, max(int(max_sleep_ms), 1)) / 1000.0)


def is_full_consecutive_forecast_cycle_enabled(spec: dict[str, Any]) -> bool:
    return (
        str(spec.get("family", "")) == "consecutive_rollout"
        and int(spec.get("pred_len", 1)) > 1
        and bool(spec.get("use_full_forecast_cycle", False))
    )


def is_fast_1m_full_forecast_cycle_enabled(config: dict[str, Any]) -> bool:
    spec = config["models"].get("1m", {})
    configured = bool(config.get("execution", {}).get("fast_1m_use_full_forecast_cycle", False))
    return is_fast_1m_enabled(config) and (configured or is_full_consecutive_forecast_cycle_enabled(spec))


def build_consecutive_forecast_cycle(spec: dict[str, Any], market_path: Path) -> dict[str, Any]:
    context_df, forecast_df = generate_consecutive_forecast(spec, market_path)
    context_timestamp = pd.Timestamp(context_df["timestamp"].iloc[-1])
    rows: list[dict[str, Any]] = []
    for row in forecast_df.itertuples(index=False):
        rows.append(
            {
                "timestamp": str(pd.Timestamp(row.timestamp).isoformat()),
                "pred_dir": int(row.pred_dir),
                "move_pct": None if pd.isna(row.move_pct) else float(row.move_pct),
                "reference_close": None if pd.isna(row.reference_close) else float(row.reference_close),
                "close": float(row.close),
            }
        )
    return {
        "metric_name": str(spec.get("metric_name", "")),
        "context_timestamp": str(context_timestamp.isoformat()),
        "binance_interval": str(spec["binance_interval"]),
        "rows": rows,
    }


def build_fast_1m_forecast_cycle(spec: dict[str, Any], market_path: Path) -> dict[str, Any]:
    return build_consecutive_forecast_cycle(spec, market_path)


def parse_iso_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts


def compute_consecutive_forecast_step_index(
    cycle_context_timestamp: str | None,
    current_context_timestamp: pd.Timestamp,
    interval: str,
) -> int | None:
    base_ts = parse_iso_timestamp(cycle_context_timestamp)
    if base_ts is None:
        return None
    step = pd.Timedelta(milliseconds=INTERVAL_TO_MS[str(interval)])
    delta = current_context_timestamp - base_ts
    if delta < pd.Timedelta(0):
        return None
    if delta % step != pd.Timedelta(0):
        return None
    return int(delta / step)


def compute_fast_1m_forecast_step_index(
    cycle_context_timestamp: str | None,
    current_context_timestamp: pd.Timestamp,
    interval: str,
) -> int | None:
    return compute_consecutive_forecast_step_index(
        cycle_context_timestamp=cycle_context_timestamp,
        current_context_timestamp=current_context_timestamp,
        interval=interval,
    )


def build_consecutive_signal_from_cycle(
    spec: dict[str, Any],
    cycle: dict[str, Any],
    step_index: int,
    current_context_timestamp: pd.Timestamp,
    market_path: Path,
) -> ModelSignal:
    rows = list(cycle.get("rows") or [])
    if step_index < 0 or step_index >= len(rows):
        raise IndexError(f"Forecast step {step_index} out of range for cycle with {len(rows)} rows")
    row = pd.Series(
        {
            "timestamp": parse_iso_timestamp(str(rows[step_index].get("timestamp"))),
            "pred_dir": int(rows[step_index].get("pred_dir", 0)),
            "move_pct": rows[step_index].get("move_pct"),
            "reference_close": rows[step_index].get("reference_close"),
            "close": rows[step_index].get("close"),
        }
    )
    return build_consecutive_signal_from_row(
        spec=spec,
        context_timestamp=current_context_timestamp,
        market_path=market_path,
        row=row,
    )


def build_fast_1m_signal_from_cycle(
    spec: dict[str, Any],
    cycle: dict[str, Any],
    step_index: int,
    current_context_timestamp: pd.Timestamp,
    market_path: Path,
) -> ModelSignal:
    return build_consecutive_signal_from_cycle(
        spec=spec,
        cycle=cycle,
        step_index=step_index,
        current_context_timestamp=current_context_timestamp,
        market_path=market_path,
    )


def resolve_interval_forecast_cycle_signal(
    *,
    state: dict[str, Any],
    state_key: str,
    spec: dict[str, Any],
    market_path: Path,
) -> tuple[ModelSignal, dict[str, Any]]:
    current_market_df = load_market_dataframe(market_path)
    current_context_timestamp = pd.Timestamp(current_market_df["timestamp"].iloc[-1])
    cycle = state.get(state_key)
    refreshed = False

    if not isinstance(cycle, dict):
        cycle = None
    if isinstance(cycle, dict):
        if str(cycle.get("metric_name", "")) != str(spec.get("metric_name", "")):
            cycle = None
        elif str(cycle.get("binance_interval", "")) != str(spec.get("binance_interval", "")):
            cycle = None

    step_index = None
    step_count = None
    if isinstance(cycle, dict):
        step_index = compute_consecutive_forecast_step_index(
            cycle_context_timestamp=cycle.get("context_timestamp"),
            current_context_timestamp=current_context_timestamp,
            interval=str(spec["binance_interval"]),
        )
        step_count = len(list(cycle.get("rows") or []))
        if (
            step_index is None
            or step_count <= 0
            or step_index < 0
            or step_index >= step_count
        ):
            cycle = None

    if not isinstance(cycle, dict):
        cycle = build_consecutive_forecast_cycle(spec, market_path)
        state[state_key] = cycle
        refreshed = True
        step_index = compute_consecutive_forecast_step_index(
            cycle_context_timestamp=cycle.get("context_timestamp"),
            current_context_timestamp=current_context_timestamp,
            interval=str(spec["binance_interval"]),
        )
        step_count = len(list(cycle.get("rows") or []))

    cycle_context_timestamp = str(cycle.get("context_timestamp") or "")
    if (
        step_index is None
        or step_count is None
        or step_index < 0
        or step_index >= step_count
    ):
        raise ValueError(
            f"Current context {current_context_timestamp.isoformat()} is outside forecast cycle "
            f"generated from {cycle_context_timestamp}"
        )

    signal = build_consecutive_signal_from_cycle(
        spec=spec,
        cycle=cycle,
        step_index=step_index,
        current_context_timestamp=current_context_timestamp,
        market_path=market_path,
    )
    diagnostics = {
        "context_timestamp": cycle_context_timestamp,
        "refreshed": refreshed,
        "step_index": step_index,
        "step_count": step_count,
    }
    return signal, diagnostics


def run_fast_1m_cycle(
    config: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    market_path: Path,
    client: Any,
    feed: FastMinuteMarketDataFeed,
    candle_close_server_ms: int,
) -> dict[str, Any]:
    mode = str(args.mode or config["mode"]).lower()
    if mode not in {"dry_run", "testnet", "demo", "live"}:
        raise ValueError(f"Unsupported mode: {mode}")
    config["mode"] = mode

    state_path = Path(config["paths"]["state_path"])
    signal_log_path = Path(config["paths"]["signal_log_path"])
    order_log_path = Path(config["paths"]["order_log_path"])
    status_log_path = resolve_optional_runtime_path(config, "status_log_path", "status_log.jsonl")
    state = read_json(state_path, default={}) or {}

    signal_errors: dict[str, str] = {}
    signals: dict[str, ModelSignal] = {}
    spec = config["models"]["1m"]
    current_market_df = load_market_dataframe(market_path)
    current_context_timestamp = pd.Timestamp(current_market_df["timestamp"].iloc[-1])
    forecast_cycle_refreshed = False
    forecast_cycle_step_index: int | None = None
    forecast_cycle_step_count: int | None = None
    forecast_cycle_context_timestamp: str | None = None
    full_forecast_cycle_mode = is_fast_1m_full_forecast_cycle_enabled(config)
    try:
        if full_forecast_cycle_mode:
            if bool(args.force):
                state.pop("fast_1m_forecast_cycle", None)
            signals["1m"], diagnostics = resolve_interval_forecast_cycle_signal(
                state=state,
                state_key="fast_1m_forecast_cycle",
                spec=spec,
                market_path=market_path,
            )
            forecast_cycle_refreshed = bool(diagnostics["refreshed"])
            forecast_cycle_step_index = int(diagnostics["step_index"])
            forecast_cycle_step_count = int(diagnostics["step_count"])
            forecast_cycle_context_timestamp = str(diagnostics["context_timestamp"])
        else:
            signals["1m"] = generate_consecutive_signal(config, spec, market_path)
    except Exception as exc:
        signal_errors["1m"] = f"{spec['model_id']}: {exc}"

    close_detection_latency_ms = max(int(feed.server_now_ms()) - int(candle_close_server_ms), 0)
    output: dict[str, Any] = {
        "generated_at": exchange_utc_now_iso(),
        "mode": mode,
        "symbol": config["symbol"],
        "trigger": {
            "source": "fast_1m_server_clock",
            "candle_close_server_ms": int(candle_close_server_ms),
            "close_detection_latency_ms": close_detection_latency_ms,
            "inference_device": str(spec.get("device", "")),
            "full_forecast_cycle_mode": full_forecast_cycle_mode,
            "forecast_cycle_refreshed": forecast_cycle_refreshed,
            "forecast_cycle_context_timestamp": forecast_cycle_context_timestamp,
            "forecast_step_index": forecast_cycle_step_index,
            "forecast_step_count": forecast_cycle_step_count,
        },
        "signals": summarize_signals(signals),
        "signal_errors": signal_errors,
        "signal_summary": None,
        "skipped": False,
        "skip_reason": None,
        "execution": None,
    }

    if signal_errors:
        output["skipped"] = True
        output["skip_reason"] = "Required model signal generation failed"
        output["execution"] = {
            "kind": "blocked",
            "reason": "signal_generation_failed",
            "required_intervals": ["1m"],
            "missing_intervals": ["1m"],
            "blocking_signal_errors": signal_errors,
            "trigger_source": "fast_1m_server_clock",
            "close_detection_latency_ms": close_detection_latency_ms,
            "full_forecast_cycle_mode": full_forecast_cycle_mode,
            "forecast_cycle_refreshed": forecast_cycle_refreshed,
            "forecast_cycle_context_timestamp": forecast_cycle_context_timestamp,
            "forecast_step_index": forecast_cycle_step_index,
            "forecast_step_count": forecast_cycle_step_count,
        }
        output["cycle_status"] = build_cycle_status(output, config)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    signal_summary = build_single_interval_strategy(config, signals)
    skip_for_same_bar, skip_reason = maybe_skip_same_primary_bar(state, config, signals, force=bool(args.force))
    current_price = feed.get_last_trade_price() or resolve_current_price(config)
    output["signals"] = summarize_signals(signals)
    output["signal_summary"] = signal_summary
    output["skipped"] = skip_for_same_bar
    output["skip_reason"] = skip_reason

    api_key = os.getenv(str(config["api"]["key_env"]), "")
    api_secret = os.getenv(str(config["api"]["secret_env"]), "")
    order_style = get_order_style(config)

    if mode == "dry_run" or not api_key or not api_secret:
        dry_run_execution: dict[str, Any] = {
            "kind": "dry_run",
            "api_keys_present": bool(api_key and api_secret),
            "order_style": order_style,
            "trigger_source": "fast_1m_server_clock",
            "close_detection_latency_ms": close_detection_latency_ms,
            "inference_device": str(spec.get("device", "")),
            "full_forecast_cycle_mode": full_forecast_cycle_mode,
            "forecast_cycle_refreshed": forecast_cycle_refreshed,
            "forecast_cycle_context_timestamp": forecast_cycle_context_timestamp,
            "forecast_step_index": forecast_cycle_step_index,
            "forecast_step_count": forecast_cycle_step_count,
            "portfolio_preview": build_dry_run_portfolio_preview(config, signal_summary, float(current_price)),
        }
        try:
            exchange_info = client.get_exchange_info(config["symbol"])
            symbol_rules = extract_symbol_rules(exchange_info, config["symbol"])
            book_ticker = client.get_book_ticker(config["symbol"]) if order_style == "maker_limit" else None
            paper_balances = build_paper_balance_map(
                symbol_rules=symbol_rules,
                paper_equity_quote=float(config["risk"]["default_paper_equity_usdt"]),
            )
            dry_run_execution["order_plan"] = plan_spot_rebalance(
                config=config,
                signal_summary=signal_summary,
                symbol_rules=symbol_rules,
                balances=paper_balances,
                current_price=current_price,
                book_ticker=book_ticker,
            )
        except Exception as exc:
            dry_run_execution["order_plan_error"] = str(exc)
        output["execution"] = dry_run_execution
        output["cycle_status"] = build_cycle_status(output, config)
        if skip_for_same_bar:
            append_jsonl(status_log_path, output["cycle_status"])
            append_jsonl(signal_log_path, output)
            return output
        state["last_primary_context_timestamp"] = current_primary_context_timestamp(signals, str(config["strategy"]["primary_interval"]))
        write_json(state_path, state)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    exchange_info = client.get_exchange_info(config["symbol"])
    symbol_rules = extract_symbol_rules(exchange_info, config["symbol"])
    account = client.get_account()
    balances = build_balance_map(account)
    book_ticker = client.get_book_ticker(config["symbol"]) if order_style == "maker_limit" else None
    rebalance_plan = plan_spot_rebalance(
        config=config,
        signal_summary=signal_summary,
        symbol_rules=symbol_rules,
        balances=balances,
        current_price=current_price,
        book_ticker=book_ticker,
    )

    state = update_risk_state(state, config, equity_quote=float(rebalance_plan["equity_quote"]))
    if state.get("risk_halt"):
        signal_summary["target_exposure"] = 0.0
        signal_summary["target_allocation"] = 0.0
        signal_summary["reasons"] = list(signal_summary.get("reasons", [])) + [str(state["risk_halt_reason"])]
        rebalance_plan = plan_spot_rebalance(
            config=config,
            signal_summary=signal_summary,
            symbol_rules=symbol_rules,
            balances=balances,
            current_price=current_price,
            book_ticker=book_ticker,
        )

    execution = {
        "kind": "spot",
        "order_style": rebalance_plan.get("order_style"),
        "account_equity_quote": rebalance_plan["equity_quote"],
        "target_exposure": signal_summary.get("target_exposure"),
        "current_allocation": rebalance_plan["current_allocation"],
        "target_allocation": rebalance_plan["target_allocation"],
        "spot_execution_clips_short": bool((signal_summary.get("target_exposure") or 0.0) < 0.0),
        "delta_quote": rebalance_plan["delta_quote"],
        "side": rebalance_plan.get("side"),
        "order_reason": rebalance_plan.get("order_reason"),
        "book_ticker": rebalance_plan.get("book_ticker"),
        "pricing_mode": rebalance_plan.get("pricing_mode"),
        "limit_price": rebalance_plan.get("limit_price"),
        "limit_quantity": rebalance_plan.get("limit_quantity"),
        "order_params": rebalance_plan["order_params"],
        "risk_halt": bool(state.get("risk_halt")),
        "risk_halt_reason": state.get("risk_halt_reason"),
        "trigger_source": "fast_1m_server_clock",
        "close_detection_latency_ms": close_detection_latency_ms,
        "fast_ws_orders": bool(config.get("execution", {}).get("fast_1m_ws_orders", False)),
        "inference_device": str(spec.get("device", "")),
        "full_forecast_cycle_mode": full_forecast_cycle_mode,
        "forecast_cycle_refreshed": forecast_cycle_refreshed,
        "forecast_cycle_context_timestamp": forecast_cycle_context_timestamp,
        "forecast_step_index": forecast_cycle_step_index,
        "forecast_step_count": forecast_cycle_step_count,
    }

    allow_same_bar_maintenance = (
        skip_for_same_bar
        and order_style == "maker_limit"
        and bool(args.send_orders)
        and not bool(args.test_order)
        and bool(config.get("execution", {}).get("allow_same_bar_order_maintenance", True))
    )
    execution["same_bar_maintenance"] = allow_same_bar_maintenance

    if skip_for_same_bar and not allow_same_bar_maintenance:
        try:
            execution, state = enrich_execution_with_live_status(
                client=client,
                config=config,
                state=state,
                execution=execution,
                order_log_path=order_log_path,
            )
        except Exception as exc:
            execution["status_error"] = str(exc)
        output["execution"] = execution
        output["cycle_status"] = build_cycle_status(output, config)
        write_json(state_path, state)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    should_execute_maker = bool(args.send_orders) and order_style == "maker_limit"
    should_execute_market = bool(args.send_orders) and order_style != "maker_limit" and bool(rebalance_plan["order_params"])

    if should_execute_maker:
        maker_execution = execute_maker_limit_plan(
            client=client,
            config=config,
            plan=rebalance_plan,
            test_order=bool(args.test_order),
        )
        execution.update(maker_execution)
        append_jsonl(
            order_log_path,
            {
                "generated_at": exchange_utc_now_iso(),
                "mode": mode,
                "symbol": config["symbol"],
                "trigger": output["trigger"],
                "order_params": rebalance_plan["order_params"],
                "response": execution.get("response"),
                "order_style": order_style,
                "action": execution.get("action"),
                "cancel_responses": execution.get("cancel_responses"),
            },
        )
    elif should_execute_market:
        response = client.place_order(rebalance_plan["order_params"], test_only=bool(args.test_order))
        execution["order_sent"] = True
        execution["test_order"] = bool(args.test_order)
        execution["response"] = response
        append_jsonl(
            order_log_path,
            {
                "generated_at": exchange_utc_now_iso(),
                "mode": mode,
                "symbol": config["symbol"],
                "trigger": output["trigger"],
                "order_params": rebalance_plan["order_params"],
                "response": response,
                "order_style": order_style,
                "action": "new_market_order",
            },
        )
    else:
        execution["order_sent"] = False
        execution["test_order"] = bool(args.test_order)

    try:
        execution, state = enrich_execution_with_live_status(
            client=client,
            config=config,
            state=state,
            execution=execution,
            order_log_path=order_log_path,
        )
    except Exception as exc:
        execution["status_error"] = str(exc)

    state["last_primary_context_timestamp"] = current_primary_context_timestamp(signals, str(config["strategy"]["primary_interval"]))
    write_json(state_path, state)
    output["execution"] = execution
    output["cycle_status"] = build_cycle_status(output, config)
    append_jsonl(status_log_path, output["cycle_status"])
    append_jsonl(signal_log_path, output)
    return output


def should_run_same_bar_maintenance(
    result: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not bool(args.send_orders) or bool(args.test_order):
        return False
    if get_order_style(config) != "maker_limit":
        return False
    if not bool(config.get("execution", {}).get("allow_same_bar_order_maintenance", True)):
        return False
    execution = result.get("execution") or {}
    if execution.get("kind") != "spot":
        return False
    execution_cfg = config.get("execution", {})
    require_existing_order = bool(execution_cfg.get("same_bar_maintenance_requires_existing_order", False))
    if bool(execution.get("managed_open_order_count") or 0):
        return True
    if bool(execution.get("order_sent")):
        return True
    if require_existing_order:
        return False
    if execution.get("order_params"):
        return True
    side = execution.get("side")
    order_reason = str(execution.get("order_reason") or "")
    return bool(side) and order_reason not in {"already_at_target", "rebalance_below_min_notional"}


def align_startup_state_to_current_primary_bar(
    config: dict[str, Any],
    *,
    base_url: str,
) -> bool:
    state_path = Path(config["paths"]["state_path"])
    state = read_json(state_path, default={}) or {}
    try:
        signals, signal_errors, _ = build_signal_map(config, base_url=base_url, state=state)
    except Exception:
        return False
    if signal_errors:
        return False
    primary_interval = str(config["strategy"]["primary_interval"])
    if primary_interval not in signals:
        return False
    state["last_primary_context_timestamp"] = current_primary_context_timestamp(signals, primary_interval)
    write_json(state_path, state)
    return True


def run_aligned_primary_interval_loop(config: dict[str, Any], args: argparse.Namespace) -> int:
    execution_cfg = config.get("execution", {})
    mode = str(args.mode or config["mode"]).lower()
    primary_interval = str(config["strategy"]["primary_interval"])
    spec = config["models"][primary_interval]
    interval_ms = int(INTERVAL_TO_MS[str(spec["binance_interval"])])
    close_buffer_ms = max(int(execution_cfg.get("primary_close_buffer_ms", 250)), 0)
    maintenance_sleep_ms = max(
        int(execution_cfg.get("primary_same_bar_maintenance_interval_ms", max(args.poll_seconds, 1) * 1000)),
        250,
    )
    client = None
    try:
        ensure_exchange_time_offset(resolve_base_url(mode))
    except Exception:
        pass
    api_key = os.getenv(str(config["api"]["key_env"]), "")
    api_secret = os.getenv(str(config["api"]["secret_env"]), "")
    if mode != "dry_run" and api_key and api_secret:
        client = build_live_spot_client(
            config=config,
            mode=mode,
            enable_ws_orders=bool(execution_cfg.get("ws_order_entry", False)),
        )

    def execute_once() -> dict[str, Any]:
        result = run_cycle(config=copy.deepcopy(config), args=args, client=client)
        cycle_status = result.get("cycle_status") or build_cycle_status(result, config)
        print(format_cycle_status_line(cycle_status), flush=True)
        return result

    try:
        startup_result: dict[str, Any] | None = None
        if not bool(args.force):
            now_ms = int(exchange_utc_now_ms())
            most_recent_close_exchange_ms = (now_ms // interval_ms) * interval_ms
            first_valid_run_exchange_ms = most_recent_close_exchange_ms + close_buffer_ms
            if now_ms < first_valid_run_exchange_ms:
                sleep_until_exchange_time(first_valid_run_exchange_ms)
            else:
                # Mid-bar restarts should not create a fresh signal-driven order.
                # They may resume repricing an already-open maker order for the
                # current bar, but otherwise wait until the next aligned close.
                if align_startup_state_to_current_primary_bar(config, base_url=resolve_base_url(mode)):
                    startup_result = execute_once()
        while True:
            try:
                result = startup_result if startup_result is not None else execute_once()
                startup_result = None
                next_close_exchange_ms = ((int(exchange_utc_now_ms()) // interval_ms) + 1) * interval_ms
                while should_run_same_bar_maintenance(result=result, config=config, args=args):
                    now_ms = int(exchange_utc_now_ms())
                    maintenance_deadline_ms = next_close_exchange_ms - close_buffer_ms
                    if now_ms >= maintenance_deadline_ms:
                        break
                    sleep_until_exchange_time(
                        min(now_ms + maintenance_sleep_ms, maintenance_deadline_ms),
                        max_sleep_ms=min(maintenance_sleep_ms, 250),
                    )
                    if int(exchange_utc_now_ms()) >= maintenance_deadline_ms:
                        break
                    result = execute_once()
                sleep_until_exchange_time(next_close_exchange_ms + close_buffer_ms)
            except Exception as exc:  # pragma: no cover - operational guard
                error_payload = {"generated_at": exchange_utc_now_iso(), "error": str(exc)}
                print(json.dumps(error_payload, indent=2, sort_keys=True), file=sys.stderr)
                time.sleep(max(int(args.poll_seconds), 1))
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()


def run_fast_1m_loop(config: dict[str, Any], args: argparse.Namespace) -> int:
    mode = str(args.mode or config["mode"]).lower()
    stream_base_url = resolve_stream_base_url(mode)
    if stream_base_url is None:
        raise RuntimeError(f"Fast 1m loop is only supported for dry_run/live/testnet modes, not {mode}")

    base_url = resolve_base_url(mode)
    try:
        ensure_exchange_time_offset(base_url)
    except Exception:
        pass
    resolved_device = configure_fast_1m_inference(config)
    market_path, keep_rows = build_fast_1m_seed_cache(config, base_url)
    if bool(config.get("execution", {}).get("fast_1m_prewarm", True)):
        prewarm_summary = prewarm_fast_1m_inference(config, market_path)
        print(
            (
                "fast_1m_prewarm "
                f"device={prewarm_summary['device']} "
                f"seconds={prewarm_summary['seconds']} "
                f"ctx={prewarm_summary['context_timestamp']} "
                f"dir={prewarm_summary['direction']}"
            ),
            flush=True,
        )
    else:
        print(f"fast_1m_prewarm skipped device={resolved_device}", flush=True)
    feed = FastMinuteMarketDataFeed(stream_base_url=stream_base_url, base_url=base_url, symbol=str(config["symbol"]))
    feed.sync_server_time()
    feed.start()
    if not feed.wait_until_ready(timeout_seconds=20.0):
        raise RuntimeError(f"Fast 1m market data stream did not become ready for {config['symbol']}")

    client = build_live_spot_client(
        config=config,
        mode=mode,
        enable_ws_orders=bool(config.get("execution", {}).get("fast_1m_ws_orders", False)),
        book_ticker_provider=feed.get_book_ticker,
    )

    try:
        if bool(args.force):
            result = run_fast_1m_cycle(
                config=copy.deepcopy(config),
                args=args,
                base_url=base_url,
                market_path=market_path,
                client=client,
                feed=feed,
                candle_close_server_ms=int(feed.server_now_ms()),
            )
            cycle_status = result.get("cycle_status") or build_cycle_status(result, config)
            print(format_cycle_status_line(cycle_status), flush=True)

        close_buffer_ms = max(int(config.get("execution", {}).get("fast_1m_close_buffer_ms", 150)), 0)
        maintenance_interval_ms = max(int(config.get("execution", {}).get("fast_1m_same_bar_maintenance_interval_ms", 500)), 100)
        while True:
            feed.ensure_fresh_server_time()
            next_close_server_ms = ((int(feed.server_now_ms()) // 60_000) + 1) * 60_000
            sleep_until_server_time(feed, next_close_server_ms + close_buffer_ms)
            closed_bar = feed.finalize_closed_bar(next_close_server_ms)
            if closed_bar is None:
                error_payload = {
                    "generated_at": exchange_utc_now_iso(),
                    "symbol": config["symbol"],
                    "error": f"Fast 1m feed has no finalized bar at close={next_close_server_ms}",
                    "stream_error": feed.last_stream_error(),
                }
                print(json.dumps(error_payload, sort_keys=True), file=sys.stderr, flush=True)
                continue
            upsert_fast_closed_bar(market_path, closed_bar, keep_rows)
            result = run_fast_1m_cycle(
                config=copy.deepcopy(config),
                args=args,
                base_url=base_url,
                market_path=market_path,
                client=client,
                feed=feed,
                candle_close_server_ms=next_close_server_ms,
            )
            cycle_status = result.get("cycle_status") or build_cycle_status(result, config)
            print(format_cycle_status_line(cycle_status), flush=True)
            while should_run_same_bar_maintenance(result=result, config=config, args=args):
                maintenance_deadline_ms = next_close_server_ms + 60_000 - close_buffer_ms
                now_ms = int(feed.server_now_ms())
                if now_ms >= maintenance_deadline_ms:
                    break
                sleep_until_server_time(
                    feed,
                    min(now_ms + maintenance_interval_ms, maintenance_deadline_ms),
                    max_sleep_ms=min(maintenance_interval_ms, 250),
                )
                if int(feed.server_now_ms()) >= maintenance_deadline_ms:
                    break
                result = run_fast_1m_cycle(
                    config=copy.deepcopy(config),
                    args=args,
                    base_url=base_url,
                    market_path=market_path,
                    client=client,
                    feed=feed,
                    candle_close_server_ms=next_close_server_ms,
                )
                cycle_status = result.get("cycle_status") or build_cycle_status(result, config)
                print(format_cycle_status_line(cycle_status), flush=True)
    finally:
        client.close()
        feed.close()


def run_cycle(
    config: dict[str, Any],
    args: argparse.Namespace,
    client: HybridBinanceSpotClient | BinanceSpotClient | None = None,
) -> dict[str, Any]:
    mode = str(args.mode or config["mode"]).lower()
    if mode not in {"dry_run", "testnet", "demo", "live"}:
        raise ValueError(f"Unsupported mode: {mode}")
    config["mode"] = mode

    state_path = Path(config["paths"]["state_path"])
    signal_log_path = Path(config["paths"]["signal_log_path"])
    order_log_path = Path(config["paths"]["order_log_path"])
    status_log_path = resolve_optional_runtime_path(config, "status_log_path", "status_log.jsonl")
    state = read_json(state_path, default={}) or {}

    base_url = resolve_base_url(mode)
    try:
        ensure_exchange_time_offset(base_url)
    except Exception:
        pass
    signals, signal_errors, forecast_cycle_diagnostics = build_signal_map(config, base_url=base_url, state=state)
    output: dict[str, Any] = {
        "generated_at": exchange_utc_now_iso(),
        "mode": mode,
        "symbol": config["symbol"],
        "signals": summarize_signals(signals),
        "signal_errors": signal_errors,
        "forecast_cycles": forecast_cycle_diagnostics,
        "signal_summary": None,
        "skipped": False,
        "skip_reason": None,
        "execution": None,
    }

    active_model_keys = get_active_model_keys(config)
    if get_strategy_mode(config) == "single_interval":
        required_intervals = set(active_model_keys)
    else:
        required_intervals = {"1d", "4h", "15m"}
    missing_intervals = [interval for interval in active_model_keys if interval not in signals]
    blocking_signal_errors = {
        interval: signal_errors[interval]
        for interval in required_intervals
        if interval in signal_errors
    }
    if blocking_signal_errors:
        output["skipped"] = True
        output["skip_reason"] = "Required model signal generation failed"
        output["execution"] = {
            "kind": "blocked",
            "reason": "signal_generation_failed",
            "required_intervals": sorted(required_intervals),
            "missing_intervals": missing_intervals,
            "blocking_signal_errors": blocking_signal_errors,
        }
        output["cycle_status"] = build_cycle_status(output, config)
        write_json(state_path, state)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    if get_strategy_mode(config) != "single_interval":
        for optional_interval in ("1h", "1m"):
            if optional_interval not in signals:
                spec = config["models"][optional_interval]
                signals[optional_interval] = ModelSignal(
                    model_id=str(spec["model_id"]),
                    model_name=str(spec["model_name"]),
                    family=str(spec["family"]),
                    role=str(spec["role"]),
                    metric_name=str(spec["metric_name"]),
                    interval=str(spec["interval"]),
                    binance_interval=str(spec["binance_interval"]),
                    context_timestamp="",
                    target_timestamp="",
                    direction="FLAT",
                    signal_value=0.0,
                    score=0.0,
                    confidence_pct=None,
                    move_pct=None,
                    prob_up=None,
                    reference_price=None,
                    predicted_price=None,
                    recent_vol_pct=None,
                    market_path="",
                )

    signal_summary = build_synchronized_strategy(config, signals)
    skip_for_same_bar, skip_reason = maybe_skip_same_primary_bar(state, config, signals, force=bool(args.force))
    current_price = resolve_current_price(config)
    output["signals"] = summarize_signals(signals)
    output["signal_summary"] = signal_summary
    output["skipped"] = skip_for_same_bar
    output["skip_reason"] = skip_reason

    api_key = os.getenv(str(config["api"]["key_env"]), "")
    api_secret = os.getenv(str(config["api"]["secret_env"]), "")

    order_style = get_order_style(config)
    if mode == "dry_run" or not api_key or not api_secret:
        dry_run_execution: dict[str, Any] = {
            "kind": "dry_run",
            "api_keys_present": bool(api_key and api_secret),
            "order_style": order_style,
            "portfolio_preview": build_dry_run_portfolio_preview(config, signal_summary, float(current_price)),
        }
        try:
            preview_client = BinanceSpotClient(
                base_url=base_url,
                api_key="",
                api_secret="",
                timeout_seconds=int(config["api"]["timeout_seconds"]),
            )
            exchange_info = preview_client.get_exchange_info(config["symbol"])
            symbol_rules = extract_symbol_rules(exchange_info, config["symbol"])
            book_ticker = preview_client.get_book_ticker(config["symbol"]) if order_style == "maker_limit" else None
            paper_balances = build_paper_balance_map(
                symbol_rules=symbol_rules,
                paper_equity_quote=float(config["risk"]["default_paper_equity_usdt"]),
            )
            dry_run_execution["order_plan"] = plan_spot_rebalance(
                config=config,
                signal_summary=signal_summary,
                symbol_rules=symbol_rules,
                balances=paper_balances,
                current_price=current_price,
                book_ticker=book_ticker,
            )
        except Exception as exc:
            dry_run_execution["order_plan_error"] = str(exc)
        output["execution"] = dry_run_execution
        output["cycle_status"] = build_cycle_status(output, config)
        if skip_for_same_bar:
            write_json(state_path, state)
            append_jsonl(status_log_path, output["cycle_status"])
            append_jsonl(signal_log_path, output)
            return output
        state["last_primary_context_timestamp"] = current_primary_context_timestamp(signals, str(config["strategy"]["primary_interval"]))
        write_json(state_path, state)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    if client is None:
        client = build_live_spot_client(config=config, mode=mode, enable_ws_orders=False)
    exchange_info = client.get_exchange_info(config["symbol"])
    symbol_rules = extract_symbol_rules(exchange_info, config["symbol"])
    account = client.get_account()
    balances = build_balance_map(account)
    book_ticker = client.get_book_ticker(config["symbol"]) if order_style == "maker_limit" else None
    rebalance_plan = plan_spot_rebalance(
        config=config,
        signal_summary=signal_summary,
        symbol_rules=symbol_rules,
        balances=balances,
        current_price=current_price,
        book_ticker=book_ticker,
    )

    state = update_risk_state(state, config, equity_quote=float(rebalance_plan["equity_quote"]))
    if state.get("risk_halt"):
        signal_summary["target_exposure"] = 0.0
        signal_summary["target_allocation"] = 0.0
        signal_summary["reasons"] = list(signal_summary.get("reasons", [])) + [str(state["risk_halt_reason"])]
        rebalance_plan = plan_spot_rebalance(
            config=config,
            signal_summary=signal_summary,
            symbol_rules=symbol_rules,
            balances=balances,
            current_price=current_price,
            book_ticker=book_ticker,
        )

    execution = {
        "kind": "spot",
        "order_style": rebalance_plan.get("order_style"),
        "account_equity_quote": rebalance_plan["equity_quote"],
        "target_exposure": signal_summary.get("target_exposure"),
        "current_allocation": rebalance_plan["current_allocation"],
        "target_allocation": rebalance_plan["target_allocation"],
        "spot_execution_clips_short": bool((signal_summary.get("target_exposure") or 0.0) < 0.0),
        "delta_quote": rebalance_plan["delta_quote"],
        "side": rebalance_plan.get("side"),
        "order_reason": rebalance_plan.get("order_reason"),
        "book_ticker": rebalance_plan.get("book_ticker"),
        "pricing_mode": rebalance_plan.get("pricing_mode"),
        "limit_price": rebalance_plan.get("limit_price"),
        "limit_quantity": rebalance_plan.get("limit_quantity"),
        "order_params": rebalance_plan["order_params"],
        "risk_halt": bool(state.get("risk_halt")),
        "risk_halt_reason": state.get("risk_halt_reason"),
    }

    same_bar_requires_existing_order = bool(
        config.get("execution", {}).get("same_bar_maintenance_requires_existing_order", False)
    )
    same_bar_existing_order_count = 0
    if (
        skip_for_same_bar
        and same_bar_requires_existing_order
        and order_style == "maker_limit"
        and bool(args.send_orders)
        and not bool(args.test_order)
    ):
        try:
            recv_window = str(config["api"]["recv_window_ms"])
            prefix = str(config.get("execution", {}).get("maker_client_order_prefix", "kronosmkr"))
            live_open_orders = client.get_open_orders({"symbol": str(config["symbol"]), "recvWindow": recv_window})
            same_bar_existing_order_count = len(filter_managed_orders(live_open_orders, prefix))
        except Exception:
            same_bar_existing_order_count = 0

    allow_same_bar_maintenance = (
        skip_for_same_bar
        and order_style == "maker_limit"
        and bool(args.send_orders)
        and not bool(args.test_order)
        and bool(config.get("execution", {}).get("allow_same_bar_order_maintenance", True))
        and (not same_bar_requires_existing_order or same_bar_existing_order_count > 0)
    )
    execution["same_bar_maintenance"] = allow_same_bar_maintenance
    if skip_for_same_bar and same_bar_requires_existing_order:
        execution["same_bar_existing_order_count"] = same_bar_existing_order_count

    if skip_for_same_bar and not allow_same_bar_maintenance:
        try:
            execution, state = enrich_execution_with_live_status(
                client=client,
                config=config,
                state=state,
                execution=execution,
                order_log_path=order_log_path,
            )
        except Exception as exc:
            execution["status_error"] = str(exc)
        output["execution"] = execution
        output["cycle_status"] = build_cycle_status(output, config)
        write_json(state_path, state)
        append_jsonl(status_log_path, output["cycle_status"])
        append_jsonl(signal_log_path, output)
        return output

    should_execute_maker = bool(args.send_orders) and order_style == "maker_limit"
    should_execute_market = bool(args.send_orders) and order_style != "maker_limit" and bool(rebalance_plan["order_params"])

    if should_execute_maker:
        maker_execution = execute_maker_limit_plan(
            client=client,
            config=config,
            plan=rebalance_plan,
            test_order=bool(args.test_order),
        )
        execution.update(maker_execution)
        append_jsonl(
            order_log_path,
            {
                "generated_at": exchange_utc_now_iso(),
                "mode": mode,
                "symbol": config["symbol"],
                "order_params": rebalance_plan["order_params"],
                "response": execution.get("response"),
                "order_style": order_style,
                "action": execution.get("action"),
                "cancel_responses": execution.get("cancel_responses"),
            },
        )
    elif should_execute_market:
        response = client.place_order(rebalance_plan["order_params"], test_only=bool(args.test_order))
        execution["order_sent"] = True
        execution["test_order"] = bool(args.test_order)
        execution["response"] = response
        append_jsonl(
            order_log_path,
            {
                "generated_at": exchange_utc_now_iso(),
                "mode": mode,
                "symbol": config["symbol"],
                "order_params": rebalance_plan["order_params"],
                "response": response,
                "order_style": order_style,
                "action": "new_market_order",
            },
        )
    else:
        execution["order_sent"] = False
        execution["test_order"] = bool(args.test_order)

    try:
        execution, state = enrich_execution_with_live_status(
            client=client,
            config=config,
            state=state,
            execution=execution,
            order_log_path=order_log_path,
        )
    except Exception as exc:
        execution["status_error"] = str(exc)

    state["last_primary_context_timestamp"] = current_primary_context_timestamp(signals, str(config["strategy"]["primary_interval"]))
    write_json(state_path, state)
    output["execution"] = execution
    output["cycle_status"] = build_cycle_status(output, config)
    append_jsonl(status_log_path, output["cycle_status"])
    append_jsonl(signal_log_path, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected multi-interval Kronos strategy and optionally place Binance Spot orders. "
            "Defaults to dry-run; live and testnet execution require explicit --send-orders."
        )
    )
    parser.add_argument("--config", type=str, default="", help="Optional JSON config overlay.")
    parser.add_argument("--env-file", type=str, default="", help="Optional KEY=VALUE file for API credentials.")
    parser.add_argument("--mode", type=str, default="", choices=["dry_run", "testnet", "demo", "live"], help="Execution mode.")
    parser.add_argument("--send-orders", action="store_true", help="Actually send signed Binance Spot orders.")
    parser.add_argument("--test-order", action="store_true", help="Use POST /api/v3/order/test for new-order validation instead of creating a real order.")
    parser.add_argument("--force", action="store_true", help="Ignore the primary-bar cooldown and evaluate immediately.")
    parser.add_argument("--loop", action="store_true", help="Run continuously instead of one cycle.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Loop delay when --loop is set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file).expanduser().resolve() if args.env_file else None
    load_env_file(env_path)
    config = build_default_config()
    if args.config:
        overlay = read_json(Path(args.config).expanduser().resolve(), default={}) or {}
        deep_update(config, overlay)

    if args.loop and is_fast_1m_enabled(config):
        return run_fast_1m_loop(config=config, args=args)

    if args.loop:
        return run_aligned_primary_interval_loop(config=config, args=args)
    else:
        result = run_cycle(config=config, args=args)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
