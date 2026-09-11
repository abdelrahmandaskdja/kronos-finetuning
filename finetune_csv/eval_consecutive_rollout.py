#!/usr/bin/env python3
"""Run consecutive autoregressive rollout and directional metrics on kline CSV."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer

FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]

# Conservative numeric guards to keep long-horizon autoregressive rollout stable.
FEATURE_CLIPS = {
    "open": (-1_000_000.0, 1_000_000.0),
    "high": (-1_000_000.0, 1_000_000.0),
    "low": (-1_000_000.0, 1_000_000.0),
    "close": (-1_000_000.0, 1_000_000.0),
    "volume": (0.0, 1_000_000_000.0),
    "amount": (0.0, 1_000_000_000_000_000.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consecutive autoregressive rollout evaluation")
    parser.add_argument(
        "--train-context-path",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_2018_2025.csv",
        help="Historical CSV used to provide context before eval start",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_2026_to_now.csv",
        help="Evaluation CSV. Rollout starts at this file's first timestamp.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=str(
            REPO_ROOT
            / "finetune_csv"
            / "finetuned"
            / "BTCUSDT_kline_5min_2018_2025"
            / "basemodel"
            / "best_model"
        ),
        help="Predictor model id or local path",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=str(
            REPO_ROOT
            / "finetune_csv"
            / "finetuned"
            / "BTCUSDT_kline_5min_2018_2025"
            / "tokenizer"
            / "best_model"
        ),
        help="Tokenizer id or local path",
    )
    parser.add_argument("--lookback", type=int, default=512, help="Context window length")
    parser.add_argument("--block-len", type=int, default=512, help="Autoregressive block size")
    parser.add_argument(
        "--feedback-source",
        type=str,
        choices=("predicted", "actual"),
        default="predicted",
        help=(
            'Context update source after each predicted block: "predicted" feeds model outputs back, '
            '"actual" feeds ground-truth eval rows back.'
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help='Device, e.g. "cpu" or "cuda:0"')
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling for token generation")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling for token generation")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of sampled paths averaged per block")
    parser.add_argument(
        "--fixed-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse normalization statistics from initial context for all rollout blocks",
    )
    parser.add_argument(
        "--stabilize-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply post-generation stabilization to keep OHLCV path numerically stable",
    )
    parser.add_argument(
        "--stability-apply-to-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed stabilized outputs back into rollout context (default keeps raw model context)",
    )
    parser.add_argument(
        "--stability-quantile",
        type=float,
        default=0.995,
        help="Quantile from train data used to derive stabilization caps",
    )
    parser.add_argument(
        "--stability-multiplier",
        type=float,
        default=6.0,
        help="Multiplier applied to quantile-derived stability caps",
    )
    parser.add_argument(
        "--min-return-cap",
        type=float,
        default=0.02,
        help="Minimum absolute close-to-close return cap used in stabilization",
    )
    parser.add_argument(
        "--max-return-cap",
        type=float,
        default=1.00,
        help="Maximum absolute close-to-close return cap used in stabilization",
    )
    parser.add_argument(
        "--trade-fee-bps",
        type=float,
        default=0.0,
        help="Per-side fee applied when the long-only signal enters or exits a position, in basis points.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT
        / "finetune_csv"
        / "consecutive_pred_btcusdt_5min_2026_full_from_5min2018_2025.csv",
        help="Output CSV with predicted consecutive kline path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / "finetune_csv"
        / "eval_directional_accuracy_btcusdt_5min_2026_full_consecutive_from_5min2018_2025.json",
        help="Output JSON with metrics",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamps"] + FEATURE_COLS)
    df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"], keep="last").reset_index(drop=True)
    return df


def sanitize_numeric_frame(df: pd.DataFrame, fallback_row: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if fallback_row is not None and col in fallback_row:
            out[col] = out[col].fillna(float(fallback_row[col]))
        else:
            out[col] = out[col].fillna(0.0)
        lo, hi = FEATURE_CLIPS[col]
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def compute_regression_metrics(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {
        "feature_mae": {},
        "feature_mse": {},
        "feature_rmse": {},
    }

    for col in FEATURE_COLS:
        pred = pd.to_numeric(pred_df[col], errors="coerce").astype(float)
        actual = pd.to_numeric(actual_df[col], errors="coerce").astype(float)
        diff = pred - actual
        mse = float((diff.pow(2)).mean())
        mae = float(diff.abs().mean())
        rmse = float(np.sqrt(mse))
        metrics["feature_mae"][col] = mae
        metrics["feature_mse"][col] = mse
        metrics["feature_rmse"][col] = rmse

    ohlc_cols = ["open", "high", "low", "close"]
    ohlc_mae = [float(metrics["feature_mae"][col]) for col in ohlc_cols]
    ohlc_mse = [float(metrics["feature_mse"][col]) for col in ohlc_cols]

    metrics["ohlc_mean_mae"] = float(np.mean(ohlc_mae))
    metrics["ohlc_mean_mse"] = float(np.mean(ohlc_mse))
    metrics["ohlc_mean_rmse"] = float(np.sqrt(metrics["ohlc_mean_mse"]))
    metrics["close_mae"] = float(metrics["feature_mae"]["close"])
    metrics["close_mse"] = float(metrics["feature_mse"]["close"])
    metrics["close_rmse"] = float(metrics["feature_rmse"]["close"])
    return metrics


def weighted_accuracy(correct: pd.Series, weights: pd.Series) -> float:
    correct_np = pd.to_numeric(correct, errors="coerce").fillna(0.0).astype(float).to_numpy()
    weights_np = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).to_numpy()
    weight_sum = float(weights_np.sum())
    if weight_sum <= 0.0:
        return float(correct_np.mean())
    return float(np.average(correct_np, weights=weights_np))


def compute_long_only_trade_metrics(
    signal_up: pd.Series,
    actual_close: pd.Series,
    prev_close: pd.Series,
    *,
    fee_bps: float = 0.0,
) -> dict[str, float]:
    position = (
        pd.to_numeric(signal_up, errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0, upper=1.0)
        .reset_index(drop=True)
    )
    actual_close = pd.to_numeric(actual_close, errors="coerce").astype(float).reset_index(drop=True)
    prev_close = (
        pd.to_numeric(prev_close, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
        .reset_index(drop=True)
    )
    prev_close_safe = prev_close.abs().clip(lower=1e-6)
    raw_bar_return = ((actual_close - prev_close) / prev_close_safe).fillna(0.0)

    prev_position = pd.concat([pd.Series([0.0]), position.iloc[:-1]], ignore_index=True)
    turnover = (position - prev_position).abs()
    fee_rate = max(float(fee_bps), 0.0) / 10_000.0
    net_bar_return = position * raw_bar_return - turnover * fee_rate
    if len(net_bar_return) > 0 and position.iloc[-1] > 0.0:
        # Realize the final bar as a closed trade so end-of-window results are comparable.
        net_bar_return.iloc[-1] -= position.iloc[-1] * fee_rate

    equity_curve = (1.0 + net_bar_return).cumprod()
    if len(equity_curve) == 0:
        return {
            "fee_bps": float(fee_bps),
            "bars": 0.0,
            "trade_rate": 0.0,
            "entry_count": 0.0,
            "total_return": 0.0,
            "mean_bar_return": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "return_minus_drawdown": 0.0,
            "active_win_rate": 0.0,
        }

    running_peak = equity_curve.cummax().clip(lower=1e-12)
    drawdown = 1.0 - (equity_curve / running_peak)
    gross_profit = float(net_bar_return.clip(lower=0.0).sum())
    gross_loss = float((-net_bar_return.clip(upper=0.0)).sum())
    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    active_mask = position > 0.5
    active_returns = net_bar_return[active_mask]
    active_win_rate = float((active_returns > 0.0).mean()) if len(active_returns) > 0 else 0.0

    return {
        "fee_bps": float(fee_bps),
        "bars": float(len(net_bar_return)),
        "trade_rate": float(position.mean()),
        "entry_count": float(((position > 0.5) & (prev_position <= 0.5)).sum()),
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "mean_bar_return": float(net_bar_return.mean()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "max_drawdown": float(drawdown.max()),
        "return_minus_drawdown": float((equity_curve.iloc[-1] - 1.0) - drawdown.max()),
        "active_win_rate": active_win_rate,
    }


@dataclass
class StabilityProfile:
    return_cap: float
    gap_cap: float
    high_wick_cap: float
    low_wick_cap: float
    price_abs_cap: float
    volume_cap: float
    amount_cap: float
    price_floor: float = 1e-6


def _safe_quantile(values: pd.Series, q: float, default: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def build_stability_profile(train_df: pd.DataFrame, args: argparse.Namespace) -> StabilityProfile:
    q = float(np.clip(args.stability_quantile, 0.5, 0.9999))
    mult = float(max(args.stability_multiplier, 1.0))

    close = pd.to_numeric(train_df["close"], errors="coerce").clip(lower=1e-6)
    prev_close = close.shift(1)
    abs_ret = ((close - prev_close) / prev_close).abs().replace([np.inf, -np.inf], np.nan).dropna()
    base_ret_cap = _safe_quantile(abs_ret, q, default=0.03)
    return_cap = float(np.clip(base_ret_cap * mult, args.min_return_cap, args.max_return_cap))
    gap_cap = float(np.clip(return_cap * 0.75, args.min_return_cap, args.max_return_cap))

    oc_max = pd.concat([train_df["open"], train_df["close"]], axis=1).max(axis=1).clip(lower=1e-6)
    oc_min = pd.concat([train_df["open"], train_df["close"]], axis=1).min(axis=1).clip(lower=1e-6)
    high_wick = ((train_df["high"] - oc_max) / oc_max).clip(lower=0.0)
    low_wick = ((oc_min - train_df["low"]) / oc_min).clip(lower=0.0)
    high_wick_cap = float(np.clip(_safe_quantile(high_wick, q, default=0.02) * mult, 0.001, 1.0))
    low_wick_cap = float(np.clip(_safe_quantile(low_wick, q, default=0.02) * mult, 0.001, 1.0))

    abs_price = train_df[["open", "high", "low", "close"]].abs().stack()
    price_abs_cap = float(max(_safe_quantile(abs_price, q, default=100_000.0) * mult, 1.0))

    volume_cap = float(max(_safe_quantile(train_df["volume"], q, default=1.0) * mult, 1.0))
    amount_cap = float(max(_safe_quantile(train_df["amount"], q, default=1.0) * mult, 1.0))

    return StabilityProfile(
        return_cap=return_cap,
        gap_cap=gap_cap,
        high_wick_cap=high_wick_cap,
        low_wick_cap=low_wick_cap,
        price_abs_cap=price_abs_cap,
        volume_cap=volume_cap,
        amount_cap=amount_cap,
    )


def stabilize_pred_block(
    pred_block: pd.DataFrame,
    prev_close_seed: float,
    profile: StabilityProfile,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = pred_block.copy().reset_index(drop=True)

    prev_close = float(prev_close_seed) if np.isfinite(prev_close_seed) else 0.0

    adjusted_rows = 0
    close_clip_hits = 0
    wick_clip_hits = 0
    invalid_rows = 0

    stable_rows: list[dict[str, float | pd.Timestamp]] = []
    for _, row in out.iterrows():
        adjusted = False

        open_raw = float(row["open"]) if np.isfinite(row["open"]) else prev_close
        high_raw = float(row["high"]) if np.isfinite(row["high"]) else open_raw
        low_raw = float(row["low"]) if np.isfinite(row["low"]) else open_raw
        close_raw = float(row["close"]) if np.isfinite(row["close"]) else prev_close
        volume_raw = float(row["volume"]) if np.isfinite(row["volume"]) else 0.0
        amount_raw = float(row["amount"]) if np.isfinite(row["amount"]) else volume_raw * prev_close

        if not np.isfinite([open_raw, high_raw, low_raw, close_raw, volume_raw, amount_raw]).all():
            invalid_rows += 1
            adjusted = True

        # Absolute hard bound from training distribution.
        open_abs = float(np.clip(open_raw, -profile.price_abs_cap, profile.price_abs_cap))
        close_abs = float(np.clip(close_raw, -profile.price_abs_cap, profile.price_abs_cap))
        close_val = close_abs
        open_val = open_abs

        if not np.isclose(close_val, close_raw, rtol=0.0, atol=1e-6):
            close_clip_hits += 1
            adjusted = True
        if not np.isclose(open_val, open_raw, rtol=0.0, atol=1e-6):
            adjusted = True

        max_oc = max(open_val, close_val)
        min_oc = min(open_val, close_val)

        high_abs = float(np.clip(high_raw, -profile.price_abs_cap, profile.price_abs_cap))
        low_abs = float(np.clip(low_raw, -profile.price_abs_cap, profile.price_abs_cap))
        high_val = max(high_abs, max_oc)
        low_val = min(low_abs, min_oc)

        high_val = max(high_val, max_oc)
        low_val = min(low_val, min_oc)

        if low_val > high_val:
            low_val = min_oc
            high_val = max_oc
            wick_clip_hits += 1
            adjusted = True

        if not np.isclose(high_val, high_raw, rtol=0.0, atol=1e-6):
            wick_clip_hits += 1
            adjusted = True
        if not np.isclose(low_val, low_raw, rtol=0.0, atol=1e-6):
            wick_clip_hits += 1
            adjusted = True

        volume_val = float(np.clip(volume_raw, 0.0, profile.volume_cap))
        amount_val = float(np.clip(amount_raw, 0.0, profile.amount_cap))

        if volume_val > 0.0:
            implied_price = amount_val / volume_val if volume_val > 0 else 0.0
            ref_price = max(abs(close_val), profile.price_floor)
            if (not np.isfinite(implied_price)) or (implied_price < ref_price * 0.1) or (implied_price > ref_price * 10.0):
                amount_val = float(np.clip(volume_val * ref_price, 0.0, profile.amount_cap))
                adjusted = True
        else:
            amount_val = 0.0

        if adjusted:
            adjusted_rows += 1

        stable_rows.append(
            {
                "timestamps": row["timestamps"],
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
                "amount": amount_val,
            }
        )
        prev_close = float(close_val)

    return pd.DataFrame(stable_rows), {
        "rows_adjusted": adjusted_rows,
        "close_clip_hits": close_clip_hits,
        "wick_clip_hits": wick_clip_hits,
        "invalid_rows": invalid_rows,
    }


def align_and_validate_predictions(pred_df: pd.DataFrame, actual_df: pd.DataFrame, tol: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align predicted and actual dataframes by timestamp within tolerance.

    Returns a pair of dataframes with equal length where each row i in the returned
    dfs corresponds to the same timestamp (rounded to the actual timestamp) and
    differing timestamps within `tol` are considered matched. Rows that cannot be
    matched within `tol` are dropped and reported.
    """
    pred = pred_df.copy().reset_index(drop=True)
    act = actual_df.copy().reset_index(drop=True)

    pred["timestamps"] = pd.to_datetime(pred["timestamps"])
    act["timestamps"] = pd.to_datetime(act["timestamps"])

    # quick exact-equality fast path
    if len(pred) == len(act) and (pred["timestamps"].equals(act["timestamps"])):
        return pred.reset_index(drop=True), act.reset_index(drop=True)

    # Build nearest-index mapping from pred timestamps -> actual timestamps
    act_ts = act["timestamps"].values
    pred_ts = pred["timestamps"].values

    # use searchsorted on numpy datetime64 array
    idxs = np.searchsorted(act_ts, pred_ts)
    matched_pred_idx = []
    matched_act_idx = []

    for i, p in enumerate(pred_ts):
        candidates = []
        if idxs[i] < len(act_ts):
            candidates.append(idxs[i])
        if idxs[i] - 1 >= 0:
            candidates.append(idxs[i] - 1)

        best = None
        best_dt = None
        for c in candidates:
            delta = abs(pd.Timestamp(p) - pd.Timestamp(act_ts[c]))
            if best is None or delta < best_dt:
                best = c
                best_dt = delta

        if best is not None and best_dt <= tol:
            matched_pred_idx.append(i)
            matched_act_idx.append(best)

    if len(matched_pred_idx) == 0:
        raise RuntimeError(f"No predictions match actual timestamps within tolerance={tol}")

    # deduplicate by actual index (keep first match for each actual row)
    mapping = {}
    for p_i, a_i in zip(matched_pred_idx, matched_act_idx):
        if a_i not in mapping:
            mapping[a_i] = p_i

    kept_act_idx = sorted(mapping.keys())
    kept_pred_idx = [mapping[a] for a in kept_act_idx]

    aligned_act = act.iloc[kept_act_idx].reset_index(drop=True)
    aligned_pred = pred.iloc[kept_pred_idx].reset_index(drop=True)

    if len(aligned_pred) != len(aligned_act):
        raise RuntimeError("Alignment produced unequal lengths after deduplication")

    if len(aligned_pred) < max(len(pred), len(act)):
        print(f"Warning: reduced rows by alignment: pred={len(pred)} act={len(act)} -> aligned={len(aligned_pred)}")

    return aligned_pred, aligned_act


def main() -> None:
    args = parse_args()

    if args.lookback <= 0:
        raise ValueError("--lookback must be > 0")
    if args.block_len <= 0:
        raise ValueError("--block-len must be > 0")
    if args.feedback_source == "actual" and args.stability_apply_to_context:
        print("Warning: --stability-apply-to-context is ignored when --feedback-source=actual")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be > 0")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")
    if not (0.0 < args.stability_quantile < 1.0):
        raise ValueError("--stability-quantile must be in (0, 1)")
    if args.trade_fee_bps < 0.0:
        raise ValueError("--trade-fee-bps must be >= 0")

    train_df = load_csv(args.train_context_path)
    eval_df = load_csv(args.eval_path)

    pred_start = eval_df["timestamps"].iloc[0]
    pred_end = eval_df["timestamps"].iloc[-1]

    context_df = train_df[train_df["timestamps"] < pred_start].tail(args.lookback).copy().reset_index(drop=True)
    if len(context_df) < args.lookback:
        raise ValueError(f"Need {args.lookback} context rows before {pred_start}, got {len(context_df)}")

    norm_stats = None
    if args.fixed_normalization:
        context_values = context_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        norm_stats = (context_values.mean(axis=0), context_values.std(axis=0))
        print("Using fixed normalization statistics from initial context window")

    stability_profile = None
    stability_totals = {
        "rows_adjusted": 0,
        "close_clip_hits": 0,
        "wick_clip_hits": 0,
        "invalid_rows": 0,
    }
    if args.stabilize_output:
        stability_profile = build_stability_profile(train_df, args)
        print(
            "Stability profile: "
            f"return_cap={stability_profile.return_cap:.4f}, "
            f"gap_cap={stability_profile.gap_cap:.4f}, "
            f"price_abs_cap={stability_profile.price_abs_cap:.2f}, "
            f"high_wick_cap={stability_profile.high_wick_cap:.4f}, "
            f"low_wick_cap={stability_profile.low_wick_cap:.4f}"
        )

    t0 = time.time()
    print(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer_id)
    print(f"Loading model: {args.model_id}")
    model = Kronos.from_pretrained(args.model_id)
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_context=args.lookback,
        clip=5,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    n = len(eval_df)
    print(
        f"Consecutive rollout rows={n} "
        f"range={pred_start} -> {pred_end} "
        f"lookback={args.lookback} block={args.block_len} "
        f"feedback={args.feedback_source}"
    )

    pred_blocks: list[pd.DataFrame] = []
    t_rollout = time.time()
    stability_prev_close = float(context_df.iloc[-1]["close"])
    for i in range(0, n, args.block_len):
        j = min(i + args.block_len, n)
        y_ts = eval_df["timestamps"].iloc[i:j].reset_index(drop=True)

        pred_block_raw = predictor.predict(
            df=context_df[FEATURE_COLS],
            x_timestamp=context_df["timestamps"],
            y_timestamp=y_ts,
            pred_len=len(y_ts),
            T=1.0,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=False,
            norm_stats=norm_stats,
        ).reset_index(drop=False)

        if "index" in pred_block_raw.columns:
            pred_block_raw = pred_block_raw.rename(columns={"index": "timestamps"})
        if "timestamps" not in pred_block_raw.columns:
            pred_block_raw.insert(0, "timestamps", y_ts.values)

        pred_block_raw["timestamps"] = pd.to_datetime(pred_block_raw["timestamps"])
        pred_block_raw = pred_block_raw[["timestamps"] + FEATURE_COLS].reset_index(drop=True)
        pred_block_raw = sanitize_numeric_frame(pred_block_raw, fallback_row=context_df.iloc[-1])

        pred_block = pred_block_raw
        block_stability_stats = None
        if args.stabilize_output and stability_profile is not None:
            pred_block, block_stability_stats = stabilize_pred_block(
                pred_block=pred_block_raw,
                prev_close_seed=stability_prev_close,
                profile=stability_profile,
            )
            for k in stability_totals:
                stability_totals[k] += int(block_stability_stats.get(k, 0))

        if len(pred_block) != len(y_ts):
            raise RuntimeError(
                f"Block output length mismatch: got {len(pred_block)} expected {len(y_ts)} at [{i}:{j}]"
            )

        pred_blocks.append(pred_block)
        if args.feedback_source == "actual":
            feedback_block = eval_df.iloc[i:j][["timestamps"] + FEATURE_COLS].reset_index(drop=True)
            feedback_block = sanitize_numeric_frame(feedback_block, fallback_row=context_df.iloc[-1])
        else:
            feedback_block = pred_block if args.stability_apply_to_context else pred_block_raw
        context_df = (
            pd.concat([context_df[["timestamps"] + FEATURE_COLS], feedback_block], ignore_index=True)
            .tail(args.lookback)
            .reset_index(drop=True)
        )
        context_df = sanitize_numeric_frame(context_df, fallback_row=context_df.iloc[-1])
        if args.stabilize_output and stability_profile is not None:
            if args.feedback_source == "actual":
                if len(feedback_block) > 0:
                    stability_prev_close = float(feedback_block["close"].iloc[-1])
            elif len(pred_block) > 0:
                stability_prev_close = float(pred_block["close"].iloc[-1])
        if block_stability_stats is None:
            print(f"block {i}:{j} done ({j}/{n}) elapsed={time.time() - t_rollout:.1f}s", flush=True)
        else:
            print(
                f"block {i}:{j} done ({j}/{n}) elapsed={time.time() - t_rollout:.1f}s "
                f"adj={block_stability_stats['rows_adjusted']} "
                f"close_clip={block_stability_stats['close_clip_hits']} "
                f"wick_clip={block_stability_stats['wick_clip_hits']}",
                flush=True,
            )

    pred_df = pd.concat(pred_blocks, ignore_index=True).sort_values("timestamps").reset_index(drop=True)
    actual_df = eval_df[["timestamps"] + FEATURE_COLS].sort_values("timestamps").reset_index(drop=True)

    # Align predicted and actual timestamps within a small tolerance to ensure
    # comparisons are made on matching rows. If alignment drops rows a warning
    # will be printed; if no matches are possible an error is raised.
    tol = pd.Timedelta(minutes=1)
    pred_df, actual_df = align_and_validate_predictions(pred_df, actual_df, tol)

    last_context_close = float(train_df[train_df["timestamps"] < pred_start].iloc[-1]["close"])

    pred_close = pred_df["close"].astype(float)
    actual_close = actual_df["close"].astype(float)
    pred_open = pred_df["open"].astype(float)
    actual_open = actual_df["open"].astype(float)

    # Primary direction rule:
    #   down if open > close, otherwise up.
    pred_up_ref = (pred_close >= pred_open).astype(int)
    actual_up_ref = (actual_close >= actual_open).astype(int)
    acc_ref = float((pred_up_ref == actual_up_ref).mean())

    # Legacy previous-close directional metric kept for comparison.
    prev_actual = pd.concat([pd.Series([last_context_close]), actual_close.iloc[:-1]], ignore_index=True)
    legacy_pred_up_prev_actual = (pred_close >= prev_actual).astype(int)
    legacy_actual_up_prev_actual = (actual_close >= prev_actual).astype(int)
    legacy_acc_prev_actual = float((legacy_pred_up_prev_actual == legacy_actual_up_prev_actual).mean())

    # Consecutive path metric uses previous predicted close as reference for prediction path.
    prev_pred = pd.concat([pd.Series([last_context_close]), pred_close.iloc[:-1]], ignore_index=True)
    pred_up_path = (pred_close >= prev_pred).astype(int)
    actual_up_path = legacy_actual_up_prev_actual
    acc_path = float((pred_up_path == actual_up_path).mean())

    candle_acc = acc_ref

    tp = int(((pred_up_ref == 1) & (actual_up_ref == 1)).sum())
    tn = int(((pred_up_ref == 0) & (actual_up_ref == 0)).sum())
    fp = int(((pred_up_ref == 1) & (actual_up_ref == 0)).sum())
    fn = int(((pred_up_ref == 0) & (actual_up_ref == 1)).sum())

    legacy_tp = int(((legacy_pred_up_prev_actual == 1) & (legacy_actual_up_prev_actual == 1)).sum())
    legacy_tn = int(((legacy_pred_up_prev_actual == 0) & (legacy_actual_up_prev_actual == 0)).sum())
    legacy_fp = int(((legacy_pred_up_prev_actual == 1) & (legacy_actual_up_prev_actual == 0)).sum())
    legacy_fn = int(((legacy_pred_up_prev_actual == 0) & (legacy_actual_up_prev_actual == 1)).sum())

    actual_up_ratio = float(actual_up_ref.mean())
    pred_up_ratio = float(pred_up_ref.mean())
    majority_baseline = float(max(actual_up_ratio, 1.0 - actual_up_ratio))
    actual_abs_return = ((actual_close - prev_actual).abs() / prev_actual.abs().clip(lower=1e-6)).astype(float)
    ref_return_weighted_acc = weighted_accuracy(pred_up_ref == actual_up_ref, actual_abs_return)
    path_return_weighted_acc = weighted_accuracy(pred_up_path == actual_up_path, actual_abs_return)
    long_only_path_metrics = compute_long_only_trade_metrics(
        signal_up=pred_up_path,
        actual_close=actual_close,
        prev_close=prev_actual,
        fee_bps=args.trade_fee_bps,
    )
    regression_metrics = compute_regression_metrics(pred_df=pred_df, actual_df=actual_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    pred_df_out = pred_df.copy()
    pred_df_out["timestamps"] = pred_df_out["timestamps"].dt.strftime("%Y/%m/%d %H:%M")
    pred_df_out.to_csv(args.output_csv, index=False)

    metrics = {
        "mode": "consecutive_chunked_autoregressive",
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "data_path": str(args.eval_path),
        "prediction_path_csv": str(args.output_csv),
        "rows": int(len(actual_df)),
        "lookback": int(args.lookback),
        "block_len": int(args.block_len),
        "feedback_source": args.feedback_source,
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "sample_count": int(args.sample_count),
        "trade_fee_bps": float(args.trade_fee_bps),
        "fixed_normalization": bool(args.fixed_normalization),
        "stability_apply_to_context": bool(args.stability_apply_to_context),
        "range_pred_start": str(pred_start),
        "range_pred_end": str(pred_end),
        "elapsed_seconds": float(time.time() - t_rollout),
        "direction_rule": "up_if_close_gte_open_else_down",
        "directional_accuracy_close_vs_prev_actual_close": acc_ref,
        "directional_accuracy_close_vs_prev_actual_close_legacy_prev_close_ref": legacy_acc_prev_actual,
        "directional_accuracy_close_vs_prev_close_consecutive_path": acc_path,
        "directional_accuracy_candle_close_vs_open": candle_acc,
        "return_weighted_directional_accuracy_close_vs_prev_actual_close": ref_return_weighted_acc,
        "return_weighted_directional_accuracy_close_vs_prev_close_consecutive_path": path_return_weighted_acc,
        "long_only_trade_close_vs_prev_close_consecutive_path": long_only_path_metrics,
        "long_only_total_return_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["total_return"]
        ),
        "long_only_max_drawdown_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["max_drawdown"]
        ),
        "long_only_return_minus_drawdown_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["return_minus_drawdown"]
        ),
        "long_only_profit_factor_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["profit_factor"]
        ),
        "long_only_trade_rate_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["trade_rate"]
        ),
        "long_only_active_win_rate_close_vs_prev_close_consecutive_path": float(
            long_only_path_metrics["active_win_rate"]
        ),
        "majority_baseline_accuracy": majority_baseline,
        "actual_up_ratio": actual_up_ratio,
        "predicted_up_ratio": pred_up_ratio,
        "mean_abs_actual_return_vs_prev_close": float(actual_abs_return.mean()),
        "confusion_close_vs_prev_actual_close": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "confusion_close_vs_prev_actual_close_legacy_prev_close_ref": {
            "tp": legacy_tp,
            "tn": legacy_tn,
            "fp": legacy_fp,
            "fn": legacy_fn,
        },
        "regression": regression_metrics,
    }

    if args.stabilize_output and stability_profile is not None:
        metrics["stability"] = {
            "enabled": True,
            "apply_to_context": bool(args.stability_apply_to_context),
            "return_cap": float(stability_profile.return_cap),
            "gap_cap": float(stability_profile.gap_cap),
            "high_wick_cap": float(stability_profile.high_wick_cap),
            "low_wick_cap": float(stability_profile.low_wick_cap),
            "price_abs_cap": float(stability_profile.price_abs_cap),
            "volume_cap": float(stability_profile.volume_cap),
            "amount_cap": float(stability_profile.amount_cap),
            **stability_totals,
        }
    else:
        metrics["stability"] = {"enabled": False}

    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved pred CSV: {args.output_csv}")
    print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()
