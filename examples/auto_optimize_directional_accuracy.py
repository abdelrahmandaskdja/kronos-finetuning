#!/usr/bin/env python3
"""Auto-search Kronos directional-accuracy settings on kline data.

This script searches:
1) Kronos inference parameters (lookback / sampling settings)
2) Post-processing decision rules (score weights / thresholds / abstain bands)

It uses a time-ordered train/val/test split and reports the best configs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class InferenceConfig:
    lookback: int
    clip: float
    T: float
    top_k: int
    top_p: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": int(self.lookback),
            "clip": float(self.clip),
            "T": float(self.T),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "sample_count": int(self.sample_count),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-optimize Kronos directional accuracy")

    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_last6m.csv",
        help="CSV path with columns: timestamps, open, high, low, close, [volume], [amount]",
    )
    parser.add_argument("--model-id", type=str, default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer-id", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--device", type=str, default=None, help='Override device, e.g. "cpu" or "cuda:0"')
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--search-windows",
        type=int,
        default=12000,
        help="Use only the latest N windows per lookback for speed (<=0 means all)",
    )
    parser.add_argument("--seed", type=int, default=123)

    # Inference grid
    parser.add_argument("--lookbacks", type=str, default="256,384,512")
    parser.add_argument("--clips", type=str, default="3,5,7")
    parser.add_argument("--temperatures", type=str, default="0.6,0.8,1.0,1.2")
    parser.add_argument("--top-ks", type=str, default="1,3,5,10,20")
    parser.add_argument("--top-ps", type=str, default="0.8,0.9,0.95,1.0")
    parser.add_argument("--sample-counts", type=str, default="1,3,5,8")
    parser.add_argument(
        "--max-inference-combos",
        type=int,
        default=250,
        help="Randomly sample this many inference combos from the full grid (<=0 means full grid)",
    )

    # Split and target
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-coverage", type=float, default=0.1)
    parser.add_argument("--target-low", type=float, default=0.70)
    parser.add_argument("--target-high", type=float, default=0.90)

    # Decision grid
    parser.add_argument("--decision-thresholds", type=str, default="-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5")
    parser.add_argument("--abstain-thresholds", type=str, default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--weight-grid-mode", type=str, choices=["coarse", "wide"], default="wide")

    # Reporting / cache
    parser.add_argument("--top-n-report", type=int, default=25)
    parser.add_argument("--ensemble-max-members", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "finetune_csv" / "search_cache")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "finetune_csv" / "directional_optimizer_report.json",
    )
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
    if df[FEATURE_COLS].isnull().values.any():
        raise ValueError("NaN found in OHLCVA columns after numeric conversion.")

    return df.sort_values("timestamps").reset_index(drop=True)


def build_inference_grid(args: argparse.Namespace) -> list[InferenceConfig]:
    lookbacks = parse_int_list(args.lookbacks)
    clips = parse_float_list(args.clips)
    temperatures = parse_float_list(args.temperatures)
    top_ks = parse_int_list(args.top_ks)
    top_ps = parse_float_list(args.top_ps)
    sample_counts = parse_int_list(args.sample_counts)

    combos = [
        InferenceConfig(
            lookback=lookback,
            clip=clip,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
        )
        for lookback, clip, T, top_k, top_p, sample_count in itertools.product(
            lookbacks, clips, temperatures, top_ks, top_ps, sample_counts
        )
    ]

    if args.max_inference_combos > 0 and len(combos) > args.max_inference_combos:
        rng = random.Random(args.seed)
        rng.shuffle(combos)
        combos = combos[: args.max_inference_combos]
    return combos


def build_weight_grid(mode: str) -> list[dict[str, Any]]:
    if mode == "coarse":
        raw = [
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, -1, 0),
            (0.5, 0.5, 0, 0),
            (0.7, 0.3, 0, 0),
            (0.3, 0.7, 0, 0),
            (0.5, 0, 0.5, 0),
            (0.7, 0, 0.3, 0),
            (0.5, 0.5, 0, 0.25),
            (0.6, 0.2, 0.2, 0),
            (1, -0.5, 0, 0),
            (1, 0, -0.5, 0),
        ]
    else:
        raw = []
        vals = (-1.0, 0.0, 1.0)
        for w in itertools.product(vals, repeat=4):
            if w == (0.0, 0.0, 0.0, 0.0):
                continue
            raw.append(w)
        raw.extend(
            [
                (0.7, 0.3, 0.0, 0.0),
                (0.3, 0.7, 0.0, 0.0),
                (0.8, 0.2, 0.0, 0.0),
                (0.2, 0.8, 0.0, 0.0),
                (0.6, 0.2, 0.2, 0.0),
                (0.5, 0.0, 0.5, 0.0),
                (0.5, 0.0, 0.0, 0.5),
                (1.0, -0.5, 0.0, 0.0),
                (1.0, 0.0, -0.5, 0.0),
                (-1.0, 0.5, 0.0, 0.0),
                (-1.0, 0.0, 0.5, 0.0),
            ]
        )

    out: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for w in raw:
        w_arr = np.array(w, dtype=np.float64)
        l1 = np.sum(np.abs(w_arr))
        if l1 <= 0:
            continue
        w_norm = tuple(np.round((w_arr / l1), 6).tolist())
        if w_norm in seen:
            continue
        seen.add(w_norm)
        out.append({"name": f"w={w_norm}", "weights": np.array(w_norm, dtype=np.float64)})
    return out


def objective_score(acc: float, cov: float, target_low: float, target_high: float) -> float:
    in_target = 1.0 if target_low <= acc <= target_high else 0.0
    # Prioritize in-target solutions, then higher accuracy, then coverage.
    return in_target * 1000.0 + acc * 100.0 + cov


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def compute_cache_key(
    *,
    data_path: Path,
    model_id: str,
    tokenizer_id: str,
    max_context: int,
    stride: int,
    anchors: np.ndarray,
    config: InferenceConfig,
) -> str:
    payload = {
        "data_path": str(data_path.resolve()),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "max_context": int(max_context),
        "stride": int(stride),
        "anchors_sha1": hashlib.sha1(anchors.tobytes()).hexdigest(),
        "config": config.to_dict(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def prepare_lookback_pack(df: pd.DataFrame, lookback: int, args: argparse.Namespace) -> dict[str, Any]:
    if lookback <= 0:
        raise ValueError("lookback must be > 0")

    anchors = np.arange(lookback, len(df), args.stride, dtype=np.int64)
    if args.search_windows > 0 and len(anchors) > args.search_windows:
        anchors = anchors[-args.search_windows :]
    if len(anchors) < 200:
        raise ValueError(
            f"Too few windows for lookback={lookback}: {len(anchors)}. "
            "Increase data size, reduce lookback, or reduce search-windows."
        )

    n = len(anchors)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    if val_end <= train_end or val_end >= n:
        raise ValueError(f"Invalid split for lookback={lookback}: n={n}, train_end={train_end}, val_end={val_end}")

    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    returns = pd.Series(close).pct_change().fillna(0.0)
    rolling_vol = returns.rolling(window=lookback, min_periods=max(10, lookback // 4)).std().to_numpy(dtype=np.float64)

    last_close = close[anchors - 1]
    actual_close = close[anchors]
    actual_open = open_[anchors]
    actual_up = (actual_close >= last_close).astype(np.int8)

    vol = rolling_vol[anchors - 1]
    finite = np.isfinite(vol)
    if finite.any():
        fallback = float(np.nanmedian(vol[finite]))
    else:
        fallback = 1e-4
    if not math.isfinite(fallback) or fallback <= 0:
        fallback = 1e-4
    vol = np.where(np.isfinite(vol), vol, fallback)
    vol = np.clip(vol, 1e-6, None)

    pack = {
        "anchors": anchors,
        "train_slice": slice(0, train_end),
        "val_slice": slice(train_end, val_end),
        "test_slice": slice(val_end, n),
        "last_close": last_close,
        "actual_open": actual_open,
        "actual_close": actual_close,
        "actual_up": actual_up,
        "vol": vol,
    }
    return pack


def run_inference_for_config(
    *,
    df: pd.DataFrame,
    pack: dict[str, Any],
    predictor: KronosPredictor,
    config: InferenceConfig,
    args: argparse.Namespace,
    cache_dir: Path,
) -> np.ndarray:
    anchors = pack["anchors"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = compute_cache_key(
        data_path=args.data_path,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        max_context=args.max_context,
        stride=args.stride,
        anchors=anchors,
        config=config,
    )
    cache_file = cache_dir / f"infer_{key}.npz"

    if cache_file.exists():
        loaded = np.load(cache_file, allow_pickle=False)
        cached_anchors = loaded["anchors"]
        preds = loaded["preds"]
        if np.array_equal(cached_anchors, anchors):
            return preds.astype(np.float32)

    combo_seed = args.seed + (stable_hash(json.dumps(config.to_dict(), sort_keys=True)) % 1_000_000)
    set_seed(combo_seed)

    preds = np.zeros((len(anchors), len(FEATURE_COLS)), dtype=np.float32)
    for start in range(0, len(anchors), args.batch_size):
        batch_anchors = anchors[start : start + args.batch_size]
        df_list: list[pd.DataFrame] = []
        x_ts_list: list[pd.Series] = []
        y_ts_list: list[pd.Series] = []
        for anchor in batch_anchors:
            context = df.iloc[anchor - config.lookback : anchor]
            target = df.iloc[anchor : anchor + 1]
            df_list.append(context[FEATURE_COLS].reset_index(drop=True))
            x_ts_list.append(context["timestamps"].reset_index(drop=True))
            y_ts_list.append(target["timestamps"].reset_index(drop=True))

        pred_list = predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=1,
            T=config.T,
            top_k=config.top_k,
            top_p=config.top_p,
            sample_count=config.sample_count,
            verbose=False,
        )
        for i, pred_df in enumerate(pred_list):
            preds[start + i] = pred_df.iloc[0][FEATURE_COLS].to_numpy(dtype=np.float32)

    np.savez_compressed(cache_file, anchors=anchors, preds=preds)
    return preds


def build_features(preds: np.ndarray, pack: dict[str, Any]) -> np.ndarray:
    eps = 1e-8
    pred_open = preds[:, 0].astype(np.float64)
    pred_high = preds[:, 1].astype(np.float64)
    pred_low = preds[:, 2].astype(np.float64)
    pred_close = preds[:, 3].astype(np.float64)

    last_close = pack["last_close"].astype(np.float64)
    price_scale = np.abs(last_close) + eps
    vol = pack["vol"].astype(np.float64)

    close_ret = (pred_close - last_close) / price_scale
    body_ret = (pred_close - pred_open) / price_scale
    mid_ret = ((pred_high + pred_low) * 0.5 - last_close) / price_scale
    signed_range = np.sign(pred_close - pred_open) * (pred_high - pred_low) / price_scale

    # Vol-normalized features make thresholds transferable across volatility regimes.
    x0 = close_ret / (vol + eps)
    x1 = body_ret / (vol + eps)
    x2 = mid_ret / (vol + eps)
    x3 = signed_range / (vol + eps)

    feat = np.stack([x0, x1, x2, x3], axis=1)
    return np.clip(feat, -50.0, 50.0)


def eval_binary_with_abstain(
    pred_up: np.ndarray,
    actual_up: np.ndarray,
    covered: np.ndarray,
    sl: slice,
) -> tuple[float, float]:
    p = pred_up[sl]
    a = actual_up[sl]
    c = covered[sl]
    total = len(p)
    if total == 0:
        return 0.0, 0.0
    covered_n = int(c.sum())
    if covered_n == 0:
        return 0.0, 0.0
    acc = float((p[c] == a[c]).mean())
    cov = covered_n / total
    return acc, cov


def search_best_decision_rule(
    *,
    features: np.ndarray,
    actual_up: np.ndarray,
    pack: dict[str, Any],
    weight_grid: list[dict[str, Any]],
    thresholds: list[float],
    abstains: list[float],
    min_coverage: float,
    target_low: float,
    target_high: float,
) -> dict[str, Any]:
    val_sl = pack["val_slice"]
    train_sl = pack["train_slice"]
    test_sl = pack["test_slice"]

    best: dict[str, Any] | None = None

    for w_item in weight_grid:
        w = w_item["weights"]
        score = features @ w

        for thr in thresholds:
            centered = score - thr
            pred_up = (centered >= 0).astype(np.int8)

            for abst in abstains:
                if abst > 0:
                    covered = np.abs(centered) >= abst
                else:
                    covered = np.ones_like(pred_up, dtype=bool)

                val_acc, val_cov = eval_binary_with_abstain(pred_up, actual_up, covered, val_sl)
                if val_cov < min_coverage:
                    continue

                obj = objective_score(val_acc, val_cov, target_low, target_high)
                train_acc, train_cov = eval_binary_with_abstain(pred_up, actual_up, covered, train_sl)
                test_acc, test_cov = eval_binary_with_abstain(pred_up, actual_up, covered, test_sl)
                in_target_val = target_low <= val_acc <= target_high

                cur = {
                    "weight_name": w_item["name"],
                    "weights": w.tolist(),
                    "threshold": float(thr),
                    "abstain": float(abst),
                    "val_accuracy": float(val_acc),
                    "val_coverage": float(val_cov),
                    "train_accuracy": float(train_acc),
                    "train_coverage": float(train_cov),
                    "test_accuracy": float(test_acc),
                    "test_coverage": float(test_cov),
                    "val_in_target": bool(in_target_val),
                    "objective": float(obj),
                    "pred_up_all": pred_up,
                    "covered_all": covered,
                }
                if best is None or cur["objective"] > best["objective"]:
                    best = cur

    if best is None:
        # Fallback to always-covered close-vs-last direction style.
        pred_up = (features[:, 0] >= 0).astype(np.int8)
        covered = np.ones_like(pred_up, dtype=bool)
        val_acc, val_cov = eval_binary_with_abstain(pred_up, actual_up, covered, val_sl)
        train_acc, train_cov = eval_binary_with_abstain(pred_up, actual_up, covered, train_sl)
        test_acc, test_cov = eval_binary_with_abstain(pred_up, actual_up, covered, test_sl)
        best = {
            "weight_name": "fallback_close_vs_last",
            "weights": [1.0, 0.0, 0.0, 0.0],
            "threshold": 0.0,
            "abstain": 0.0,
            "val_accuracy": float(val_acc),
            "val_coverage": float(val_cov),
            "train_accuracy": float(train_acc),
            "train_coverage": float(train_cov),
            "test_accuracy": float(test_acc),
            "test_coverage": float(test_cov),
            "val_in_target": bool(target_low <= val_acc <= target_high),
            "objective": float(objective_score(val_acc, val_cov, target_low, target_high)),
            "pred_up_all": pred_up,
            "covered_all": covered,
        }
    return best


def eval_ensemble_majority(
    members: list[dict[str, Any]],
    actual_up: np.ndarray,
    sl: slice,
) -> tuple[float, float]:
    if not members:
        return 0.0, 0.0
    pred_stack = np.stack([m["pred_up_all"] for m in members], axis=0)      # (M, N)
    cov_stack = np.stack([m["covered_all"] for m in members], axis=0)        # (M, N)

    vote_sum = (pred_stack * cov_stack).sum(axis=0)
    vote_cnt = cov_stack.sum(axis=0)
    covered = vote_cnt > 0
    pred_up = (vote_sum * 2 >= vote_cnt).astype(np.int8)

    p = pred_up[sl]
    a = actual_up[sl]
    c = covered[sl]
    total = len(p)
    if total == 0:
        return 0.0, 0.0
    covered_n = int(c.sum())
    if covered_n == 0:
        return 0.0, 0.0
    acc = float((p[c] == a[c]).mean())
    cov = covered_n / total
    return acc, cov


def prune_for_json(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out.pop("pred_up_all", None)
    out.pop("covered_all", None)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Require train_ratio>0, val_ratio>0, and train_ratio+val_ratio<1.")
    if args.min_coverage < 0 or args.min_coverage > 1:
        raise ValueError("min_coverage must be in [0, 1].")

    thresholds = parse_float_list(args.decision_thresholds)
    abstains = parse_float_list(args.abstain_thresholds)
    weight_grid = build_weight_grid(args.weight_grid_mode)
    infer_grid = build_inference_grid(args)
    if not infer_grid:
        raise ValueError("Inference grid is empty. Check input parameter lists.")

    print(f"Loading data: {args.data_path}")
    df = load_kline_csv(args.data_path)
    print(f"Rows: {len(df)}")
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
        clip=5.0,  # clip is passed per inference config during normalization/generation.
    )

    lookback_values = sorted({c.lookback for c in infer_grid})
    packs: dict[int, dict[str, Any]] = {}
    for lb in lookback_values:
        packs[lb] = prepare_lookback_pack(df, lb, args)
        n = len(packs[lb]["anchors"])
        tr = packs[lb]["train_slice"].stop
        va = packs[lb]["val_slice"].stop - packs[lb]["val_slice"].start
        te = packs[lb]["test_slice"].stop - packs[lb]["test_slice"].start
        print(f"lookback={lb}: windows={n}, train={tr}, val={va}, test={te}")

    results_raw: list[dict[str, Any]] = []
    pbar = tqdm(infer_grid, desc="Searching inference configs", unit="cfg")
    for cfg in pbar:
        pack = packs[cfg.lookback]
        # Update predictor runtime knobs from config.
        predictor.clip = float(cfg.clip)
        preds = run_inference_for_config(
            df=df,
            pack=pack,
            predictor=predictor,
            config=cfg,
            args=args,
            cache_dir=args.cache_dir,
        )
        features = build_features(preds, pack)
        best_rule = search_best_decision_rule(
            features=features,
            actual_up=pack["actual_up"],
            pack=pack,
            weight_grid=weight_grid,
            thresholds=thresholds,
            abstains=abstains,
            min_coverage=args.min_coverage,
            target_low=args.target_low,
            target_high=args.target_high,
        )

        row = {
            "config": cfg.to_dict(),
            **best_rule,
        }
        results_raw.append(row)
        pbar.set_postfix(
            val_acc=f"{best_rule['val_accuracy']:.4f}",
            test_acc=f"{best_rule['test_accuracy']:.4f}",
            cov=f"{best_rule['val_coverage']:.2f}",
            target_hit=str(best_rule["val_in_target"]),
        )

    # Rank by validation objective.
    results_raw.sort(key=lambda x: x["objective"], reverse=True)
    top_results = results_raw[: max(1, args.top_n_report)]

    # Optional majority-vote ensemble among top models.
    ensemble_result: dict[str, Any] | None = None
    if args.ensemble_max_members >= 2 and top_results:
        max_members = min(args.ensemble_max_members, len(top_results))
        best_ens: dict[str, Any] | None = None
        for k in range(2, max_members + 1):
            members = top_results[:k]
            lb = members[0]["config"]["lookback"]
            # Ensemble only within same lookback to ensure aligned windows.
            if any(m["config"]["lookback"] != lb for m in members):
                continue
            pack = packs[lb]
            val_acc, val_cov = eval_ensemble_majority(members, pack["actual_up"], pack["val_slice"])
            test_acc, test_cov = eval_ensemble_majority(members, pack["actual_up"], pack["test_slice"])
            train_acc, train_cov = eval_ensemble_majority(members, pack["actual_up"], pack["train_slice"])
            obj = objective_score(val_acc, val_cov, args.target_low, args.target_high)
            cur = {
                "members": k,
                "lookback": lb,
                "train_accuracy": train_acc,
                "train_coverage": train_cov,
                "val_accuracy": val_acc,
                "val_coverage": val_cov,
                "test_accuracy": test_acc,
                "test_coverage": test_cov,
                "val_in_target": args.target_low <= val_acc <= args.target_high,
                "objective": obj,
            }
            if best_ens is None or cur["objective"] > best_ens["objective"]:
                best_ens = cur
        ensemble_result = best_ens

    best = top_results[0]
    best_json = prune_for_json(best)
    top_json = [prune_for_json(r) for r in top_results]

    hit_70_90_on_val = bool(best["val_in_target"])
    hit_70_90_on_test = bool(args.target_low <= best["test_accuracy"] <= args.target_high)

    report = {
        "meta": {
            "data_path": str(args.data_path),
            "rows": int(len(df)),
            "model_id": args.model_id,
            "tokenizer_id": args.tokenizer_id,
            "device": predictor.device,
            "seed": int(args.seed),
            "search_windows": int(args.search_windows),
            "max_inference_combos": int(args.max_inference_combos),
            "inference_combos_evaluated": int(len(results_raw)),
            "weight_rules_evaluated_per_combo": int(len(weight_grid) * len(thresholds) * len(abstains)),
            "target_accuracy_range": [float(args.target_low), float(args.target_high)],
            "min_coverage": float(args.min_coverage),
        },
        "best_config": best_json,
        "best_hits_target_on_validation": hit_70_90_on_val,
        "best_hits_target_on_test": hit_70_90_on_test,
        "top_results": top_json,
        "best_ensemble": ensemble_result,
        "notes": [
            "High accuracy with low coverage can be misleading; check coverage carefully.",
            "Validation is used for tuning; test accuracy is the out-of-sample signal.",
            "No script can guarantee a 70-90% test hit rate on every market period.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBest configuration")
    print(json.dumps(best_json, indent=2))
    print(f"\nTarget hit on validation [{args.target_low:.2f}, {args.target_high:.2f}]: {hit_70_90_on_val}")
    print(f"Target hit on test       [{args.target_low:.2f}, {args.target_high:.2f}]: {hit_70_90_on_test}")
    if ensemble_result is not None:
        print("\nBest ensemble")
        print(json.dumps(ensemble_result, indent=2))
    print(f"\nSaved report to: {args.output_json}")


if __name__ == "__main__":
    main()
