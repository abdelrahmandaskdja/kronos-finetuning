#!/usr/bin/env python3
"""Evaluate Kronos directional accuracy on 5-minute kline CSV data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
REQUIRED_COLS = ["timestamps", "open", "high", "low", "close"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kronos directional accuracy on kline data")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_last6m.csv",
        help="CSV path with columns: timestamps, open, high, low, close, [volume], [amount]",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="NeoQuasar/Kronos-base",
        help='Hugging Face model id or local path, e.g. "NeoQuasar/Kronos-base"',
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
        help='Hugging Face tokenizer id or local path, e.g. "NeoQuasar/Kronos-Tokenizer-base"',
    )
    parser.add_argument("--lookback", type=int, default=512, help="Context window length")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in bars, e.g. 32 for 32x5m")
    parser.add_argument(
        "--num-windows",
        type=int,
        default=200,
        help="Number of rolling windows to evaluate from the tail (use <=0 for all windows)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride between evaluated anchors")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for predict_batch")
    parser.add_argument("--max-context", type=int, default=512, help="KronosPredictor max_context")
    parser.add_argument("--clip", type=float, default=5.0, help="KronosPredictor clip value")
    parser.add_argument("--T", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of samples per forecast")
    parser.add_argument("--device", type=str, default=None, help='Device override, e.g. "cpu" or "cuda:0"')
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output metrics JSON path")
    return parser.parse_args()


def load_kline_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[FEATURE_COLS].isnull().any().any():
        raise ValueError("Data contains NaN after numeric conversion in OHLCVA columns.")

    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def build_anchor_indices(total_rows: int, lookback: int, pred_len: int, stride: int, num_windows: int) -> list[int]:
    if pred_len <= 0:
        raise ValueError("pred_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    min_rows = lookback + pred_len
    if total_rows < min_rows:
        raise ValueError(f"Not enough rows ({total_rows}) for lookback={lookback} and pred_len={pred_len}.")

    # Anchor points to the first forecasted bar. Need full future horizon [anchor, anchor+pred_len).
    last_anchor_exclusive = total_rows - pred_len + 1
    anchors = list(range(lookback, last_anchor_exclusive, stride))
    if num_windows > 0 and len(anchors) > num_windows:
        anchors = anchors[-num_windows:]
    return anchors


def evaluate_directional_accuracy(df: pd.DataFrame, predictor: KronosPredictor, args: argparse.Namespace) -> dict:
    anchors = build_anchor_indices(
        total_rows=len(df),
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=args.stride,
        num_windows=args.num_windows,
    )
    if not anchors:
        raise ValueError("No valid evaluation windows found. Adjust lookback/stride/num-windows.")

    # Primary direction rule:
    #   down if open > close, otherwise up.
    move_hits = 0
    actual_up_count = 0
    pred_up_count = 0
    tp = tn = fp = fn = 0

    # Legacy previous-close directional metric kept for comparison.
    legacy_move_hits = 0
    legacy_actual_up_count = 0
    legacy_pred_up_count = 0
    legacy_tp = legacy_tn = legacy_fp = legacy_fn = 0

    for start in tqdm(range(0, len(anchors), args.batch_size), desc="Evaluating", unit="batch"):
        batch_anchors = anchors[start : start + args.batch_size]
        df_list: list[pd.DataFrame] = []
        x_ts_list: list[pd.Series] = []
        y_ts_list: list[pd.Series] = []

        for anchor in batch_anchors:
            context = df.iloc[anchor - args.lookback : anchor]
            target = df.iloc[anchor : anchor + args.pred_len]
            df_list.append(context[FEATURE_COLS].reset_index(drop=True))
            x_ts_list.append(context["timestamps"].reset_index(drop=True))
            y_ts_list.append(target["timestamps"].reset_index(drop=True))

        pred_list = predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=args.pred_len,
            T=args.T,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=False,
        )

        for local_idx, anchor in enumerate(batch_anchors):
            horizon_row_idx = args.pred_len - 1
            pred_row = pred_list[local_idx].iloc[horizon_row_idx]
            actual_row = df.iloc[anchor + horizon_row_idx]
            last_close = float(df.iloc[anchor - 1]["close"])

            pred_close = float(pred_row["close"])
            pred_open = float(pred_row["open"])
            actual_close = float(actual_row["close"])
            actual_open = float(actual_row["open"])

            # Primary metric: compare predicted and actual candle direction.
            pred_move_up = int(pred_close >= pred_open)
            actual_move_up = int(actual_close >= actual_open)

            # Legacy metric: compare close direction against previous actual close.
            legacy_pred_move_up = int(pred_close >= last_close)
            legacy_actual_move_up = int(actual_close >= last_close)

            move_hits += int(pred_move_up == actual_move_up)
            actual_up_count += actual_move_up
            pred_up_count += pred_move_up

            if pred_move_up == 1 and actual_move_up == 1:
                tp += 1
            elif pred_move_up == 0 and actual_move_up == 0:
                tn += 1
            elif pred_move_up == 1 and actual_move_up == 0:
                fp += 1
            else:
                fn += 1

            legacy_move_hits += int(legacy_pred_move_up == legacy_actual_move_up)
            legacy_actual_up_count += legacy_actual_move_up
            legacy_pred_up_count += legacy_pred_move_up

            if legacy_pred_move_up == 1 and legacy_actual_move_up == 1:
                legacy_tp += 1
            elif legacy_pred_move_up == 0 and legacy_actual_move_up == 0:
                legacy_tn += 1
            elif legacy_pred_move_up == 1 and legacy_actual_move_up == 0:
                legacy_fp += 1
            else:
                legacy_fn += 1

    total = len(anchors)
    move_acc = move_hits / total
    actual_up_ratio = actual_up_count / total
    pred_up_ratio = pred_up_count / total
    majority_baseline_acc = max(actual_up_ratio, 1.0 - actual_up_ratio)
    legacy_move_acc = legacy_move_hits / total
    legacy_actual_up_ratio = legacy_actual_up_count / total
    legacy_pred_up_ratio = legacy_pred_up_count / total

    return {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "data_path": str(args.data_path),
        "rows": int(len(df)),
        "lookback": int(args.lookback),
        "pred_len": int(args.pred_len),
        "num_windows": int(total),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "device": predictor.device,
        "direction_rule": "up_if_close_gte_open_else_down",
        "directional_accuracy_close_vs_last_close": float(move_acc),
        "directional_accuracy_candle_close_vs_open": float(move_acc),
        "directional_accuracy_close_vs_last_close_legacy_prev_close_ref": float(legacy_move_acc),
        "majority_baseline_accuracy": float(majority_baseline_acc),
        "actual_up_ratio": float(actual_up_ratio),
        "predicted_up_ratio": float(pred_up_ratio),
        "actual_up_ratio_legacy_prev_close_ref": float(legacy_actual_up_ratio),
        "predicted_up_ratio_legacy_prev_close_ref": float(legacy_pred_up_ratio),
        "confusion_close_vs_last_close": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "confusion_candle_close_vs_open": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "confusion_close_vs_last_close_legacy_prev_close_ref": {
            "tp": int(legacy_tp),
            "tn": int(legacy_tn),
            "fp": int(legacy_fp),
            "fn": int(legacy_fn),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_kline_csv(args.data_path)

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

    metrics = evaluate_directional_accuracy(df=df, predictor=predictor, args=args)
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
