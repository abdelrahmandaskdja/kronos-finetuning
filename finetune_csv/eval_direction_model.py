#!/usr/bin/env python3
"""Evaluate a finetuned Kronos direction model on a target CSV split.

Typical usage for full-year test:
python finetune_csv/eval_direction_model.py \
  --data-path finetune_csv/data/BTCUSDT_kline_1m_2026_full.csv \
  --checkpoint-path finetune_csv/dir_runs/best_direction_model.pt \
  --data-type test --train-ratio 0 --val-ratio 0 --test-ratio 1 \
  --output-json finetune_csv/dir_runs/eval_2026.json \
  --output-csv finetune_csv/dir_runs/pred_2026.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import KronosTokenizer
from finetune_csv.direction_dataset import DirectionKlineDataset
from finetune_csv.kronos_direction_model import build_kronos_direction_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def build_model_from_checkpoint_config(ckpt: dict, predictor_path_override: str | None = None):
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    predictor_path = (
        predictor_path_override
        or ckpt.get("predictor_path")
        or cfg.get("pretrained_predictor_path")
    )
    if not predictor_path:
        raise ValueError(
            "Could not resolve predictor path. Pass --predictor-path explicitly or use a checkpoint with predictor metadata."
        )

    head_hidden_dim = int(cfg.get("head_hidden_dim", 512))
    head_dropout = float(cfg.get("head_dropout", 0.1))
    head_use_layernorm = bool(cfg.get("head_use_layernorm", True))
    head_pooling = str(cfg.get("head_pooling", "last"))

    model = build_kronos_direction_model(
        pretrained_predictor_path=predictor_path,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        head_use_layernorm=head_use_layernorm,
        pooling=head_pooling,
    )
    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    predictor_path_override: str | None,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

    model = build_model_from_checkpoint_config(ckpt, predictor_path_override=predictor_path_override)
    state_dict = strip_module_prefix(ckpt["model_state_dict"])

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Backward compatibility: checkpoints with a direct Linear head.
        if "direction_head.weight" in state_dict and "direction_head.bias" in state_dict:
            in_dim = int(model.kronos.d_model)
            model.direction_head = nn.Linear(in_dim, 1)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            raise

    model.to(device)
    model.eval()
    return ckpt, model, missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Kronos direction model.")
    parser.add_argument("--data-path", type=str, required=True, help="CSV with OHLCV columns.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to best_direction_model.pt")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Optional tokenizer override. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--predictor-path",
        type=str,
        default="",
        help="Optional predictor override for model rebuild.",
    )
    parser.add_argument("--data-type", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--lookback-window", type=int, default=-1, help="Override lookback if >0.")
    parser.add_argument("--horizon-steps", type=int, default=-1, help="Override horizon if >0.")
    parser.add_argument("--vol-window", type=int, default=-1, help="Override vol window if >0.")
    parser.add_argument("--vol-k", type=float, default=-1.0, help="Override vol_k if >=0.")
    parser.add_argument("--min-eps", type=float, default=-1.0, help="Override min_eps if >=0.")
    parser.add_argument(
        "--mask-ambiguous",
        type=str,
        choices=["checkpoint", "true", "false"],
        default="checkpoint",
        help="Use checkpoint setting, or force masking on/off during evaluation.",
    )
    parser.add_argument("--clip", type=float, default=-1.0, help="Override clip if >0.")
    parser.add_argument("--train-ratio", type=float, default=-1.0, help="Override train ratio if >=0.")
    parser.add_argument("--val-ratio", type=float, default=-1.0, help="Override val ratio if >=0.")
    parser.add_argument("--test-ratio", type=float, default=-1.0, help="Override test ratio if >=0.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else None

    ckpt, model, missing_keys, unexpected_keys = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=device,
        predictor_path_override=(args.predictor_path or None),
    )

    cfg = ckpt.get("config", {})
    lookback = int(args.lookback_window if args.lookback_window > 0 else cfg.get("lookback_window", 400))
    horizon = int(args.horizon_steps if args.horizon_steps > 0 else cfg.get("horizon_steps", 1))
    vol_window = int(args.vol_window if args.vol_window > 0 else cfg.get("vol_window", 288))
    vol_k = float(args.vol_k if args.vol_k >= 0 else cfg.get("vol_k", 0.25))
    min_eps = float(args.min_eps if args.min_eps >= 0 else cfg.get("min_eps", 0.0))
    if args.mask_ambiguous == "checkpoint":
        mask_ambiguous = bool(cfg.get("mask_ambiguous", True))
    else:
        mask_ambiguous = args.mask_ambiguous == "true"
    clip = float(args.clip if args.clip > 0 else cfg.get("clip", 5.0))
    train_ratio = float(args.train_ratio if args.train_ratio >= 0 else cfg.get("train_ratio", 0.7))
    val_ratio = float(args.val_ratio if args.val_ratio >= 0 else cfg.get("val_ratio", 0.15))
    test_ratio = float(args.test_ratio if args.test_ratio >= 0 else (1.0 - train_ratio - val_ratio))

    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("Invalid split ratios; sum must be > 0.")

    tokenizer_path = args.tokenizer_path or ckpt.get("tokenizer_path") or cfg.get("pretrained_tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Could not resolve tokenizer path. Pass --tokenizer-path explicitly.")

    dataset = DirectionKlineDataset(
        data_path=args.data_path,
        data_type=args.data_type,
        lookback_window=lookback,
        horizon_steps=horizon,
        clip=clip,
        vol_window=vol_window,
        vol_k=vol_k,
        min_eps=min_eps,
        mask_ambiguous=mask_ambiguous,
        seed=args.seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.to(device)
    tokenizer.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    records = []

    total_loss_weighted = 0.0
    total_valid = 0
    tp = tn = fp = fn = 0
    cursor = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_stamp = batch["x_stamp"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            rets = batch["ret"].to(device)

            token_s1, token_s2 = tokenizer.encode(x, half=True)
            logits = model(token_s1, token_s2, x_stamp)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            valid = masks > 0.5

            loss_raw = criterion(logits, labels)
            if valid.sum().item() > 0:
                loss = (loss_raw * masks).sum() / masks.sum().clamp(min=1.0)
                batch_valid = int(valid.sum().item())
                total_loss_weighted += float(loss.item()) * batch_valid
                total_valid += batch_valid

                tp += int(((preds == 1) & (labels == 1) & valid).sum().item())
                tn += int(((preds == 0) & (labels == 0) & valid).sum().item())
                fp += int(((preds == 1) & (labels == 0) & valid).sum().item())
                fn += int(((preds == 0) & (labels == 1) & valid).sum().item())

            batch_size = labels.shape[0]
            idx = np.arange(cursor, cursor + batch_size, dtype=np.int64)
            anchor = idx + dataset.lookback_window - 1
            ts = dataset.timestamps.iloc[anchor].astype(str).tolist()

            probs_cpu = probs.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            masks_cpu = masks.detach().cpu().numpy()
            rets_cpu = rets.detach().cpu().numpy()

            for i in range(batch_size):
                records.append(
                    {
                        "timestamp": ts[i],
                        "mask": float(masks_cpu[i]),
                        "label": float(labels_cpu[i]),
                        "pred_prob_up": float(probs_cpu[i]),
                        "pred_dir": int(preds_cpu[i]),
                        "ret_horizon": float(rets_cpu[i]),
                    }
                )
            cursor += batch_size

    if total_valid <= 0:
        metrics = {
            "error": "No scorable samples in evaluation set.",
            "rows": int(len(dataset)),
            "rows_valid": 0,
            "mask_ambiguous": bool(mask_ambiguous),
        }
    else:
        da = float((tp + tn) / total_valid)
        actual_up_ratio = float((tp + fn) / total_valid)
        pred_up_ratio = float((tp + fp) / total_valid)
        majority_baseline = float(max(actual_up_ratio, 1.0 - actual_up_ratio))
        avg_loss = float(total_loss_weighted / total_valid)

        metrics = {
            "mode": "direction_eval",
            "checkpoint_path": str(checkpoint_path),
            "data_path": str(Path(args.data_path).expanduser().resolve()),
            "data_type": args.data_type,
            "rows": int(len(dataset)),
            "rows_valid": int(total_valid),
            "mask_ambiguous": bool(mask_ambiguous),
            "directional_accuracy": da,
            "avg_bce_loss": avg_loss,
            "majority_baseline_accuracy": majority_baseline,
            "actual_up_ratio": actual_up_ratio,
            "predicted_up_ratio": pred_up_ratio,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "lookback_window": int(lookback),
            "horizon_steps": int(horizon),
            "vol_window": int(vol_window),
            "vol_k": float(vol_k),
            "min_eps": float(min_eps),
            "clip": float(clip),
            "split": {
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                "test_ratio": float(test_ratio),
            },
            "device": str(device),
            "checkpoint_load": {
                "missing_keys": list(missing_keys),
                "unexpected_keys": list(unexpected_keys),
            },
        }
        if mask_ambiguous:
            metrics["directional_accuracy_masked"] = da
            metrics["avg_bce_loss_masked"] = avg_loss
        else:
            metrics["directional_accuracy_unmasked"] = da
            metrics["avg_bce_loss_unmasked"] = avg_loss

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {output_json}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(output_csv, index=False)
        print(f"Saved predictions: {output_csv}")


if __name__ == "__main__":
    main()
