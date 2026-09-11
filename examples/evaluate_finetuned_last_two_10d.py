#!/usr/bin/env python3
"""Evaluate a fine-tuned Kronos model on the previous 10 days and latest 10 days."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer
from examples.evaluate_directional_accuracy import (
    evaluate_directional_accuracy,
    load_kline_csv,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Kronos on previous 10 days vs latest 10 days"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_last6m.csv",
        help="CSV path with timestamps/OHLCV columns",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=str(
            REPO_ROOT
            / "finetune_csv"
            / "finetuned"
            / "BTCUSDT_kline_5min_last6m"
            / "basemodel"
            / "best_model"
        ),
        help="Fine-tuned model local path",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=str(
            REPO_ROOT
            / "finetune_csv"
            / "finetuned"
            / "BTCUSDT_kline_5min_last6m"
            / "tokenizer"
            / "best_model"
        ),
        help="Fine-tuned tokenizer local path",
    )
    parser.add_argument("--window-days", type=int, default=10, help="Days per window")
    parser.add_argument("--lookback", type=int, default=512, help="Context window length")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon")
    parser.add_argument(
        "--num-windows",
        type=int,
        default=0,
        help="Windows per split to evaluate (<=0 means all)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride between anchors")
    parser.add_argument("--batch-size", type=int, default=64, help="Predict batch size")
    parser.add_argument("--max-context", type=int, default=512, help="Predictor max_context")
    parser.add_argument("--clip", type=float, default=5.0, help="Predictor clip")
    parser.add_argument("--T", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--sample-count", type=int, default=1, help="Forecast samples per window")
    parser.add_argument("--device", type=str, default="cuda:0", help='Device, e.g. "cuda:0" or "cpu"')
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def split_two_windows(df: pd.DataFrame, window_days: int):
    if window_days <= 0:
        raise ValueError("window_days must be > 0")
    end_ts = df["timestamps"].max()
    split_ts = end_ts - pd.Timedelta(days=window_days)
    start_ts = end_ts - pd.Timedelta(days=window_days * 2)
    prev_df = df[(df["timestamps"] > start_ts) & (df["timestamps"] <= split_ts)].reset_index(drop=True)
    last_df = df[df["timestamps"] > split_ts].reset_index(drop=True)
    return prev_df, last_df, start_ts, split_ts, end_ts


def evaluate_period(
    label: str,
    df_period: pd.DataFrame,
    predictor: KronosPredictor,
    args: argparse.Namespace,
    start_ts,
    end_ts,
) -> dict:
    if len(df_period) <= args.lookback:
        raise ValueError(
            f"{label} has only {len(df_period)} rows, need > lookback={args.lookback}."
        )
    period_args = argparse.Namespace(**vars(args))
    period_args.data_path = Path(f"{args.data_path}::{label}")
    metrics = evaluate_directional_accuracy(df=df_period, predictor=predictor, args=period_args)
    metrics["period_label"] = label
    metrics["period_start"] = str(start_ts)
    metrics["period_end"] = str(end_ts)
    metrics["rows"] = int(len(df_period))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_kline_csv(args.data_path)
    prev_df, last_df, start_ts, split_ts, end_ts = split_two_windows(df, args.window_days)

    print(f"Source range: {df['timestamps'].min()} -> {df['timestamps'].max()} ({len(df)} rows)")
    print(
        f"Previous {args.window_days} days: {prev_df['timestamps'].min()} -> "
        f"{prev_df['timestamps'].max()} ({len(prev_df)} rows)"
    )
    print(
        f"Latest {args.window_days} days: {last_df['timestamps'].min()} -> "
        f"{last_df['timestamps'].max()} ({len(last_df)} rows)"
    )

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
        max_context=args.max_context,
        clip=args.clip,
    )

    prev_metrics = evaluate_period(
        label=f"prev_{args.window_days}d",
        df_period=prev_df,
        predictor=predictor,
        args=args,
        start_ts=start_ts,
        end_ts=split_ts,
    )
    last_metrics = evaluate_period(
        label=f"last_{args.window_days}d",
        df_period=last_df,
        predictor=predictor,
        args=args,
        start_ts=split_ts,
        end_ts=end_ts,
    )

    result = {
        "source_data_path": str(args.data_path),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "device": predictor.device,
        "window_days": int(args.window_days),
        "source_rows": int(len(df)),
        "source_start": str(df["timestamps"].min()),
        "source_end": str(df["timestamps"].max()),
        f"previous_{args.window_days}d": prev_metrics,
        f"latest_{args.window_days}d": last_metrics,
    }
    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()

