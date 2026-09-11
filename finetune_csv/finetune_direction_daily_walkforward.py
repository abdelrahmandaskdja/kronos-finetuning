"""Walk-forward 1-day direction finetuning for Kronos on BTCUSDT 5m data.

This script fine-tunes KronosDirectionModel with DirectionKlineDataset using
year-based walk-forward splits. It trains on years
[train_start_year, train_end_year] and evaluates on test_year.
Direction accuracy (MDA) is computed only on non-ambiguous samples (mask == 1).
"""

import os
import argparse
import tempfile
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import KronosTokenizer
from finetune_csv.direction_dataset import DirectionKlineDataset
from finetune_csv.kronos_direction_model import build_kronos_direction_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model,
    tokenizer,
    loader,
    optimizer,
    criterion,
    device,
    log_interval: int = 200,
):
    model.train()
    total_loss = 0.0
    total_valid = 0

    num_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)  # (B, L, 6)
        x_stamp = batch["x_stamp"].to(device)  # (B, L, 5)
        labels = batch["label"].to(device)  # (B,)
        masks = batch["mask"].to(device)  # (B,)

        valid_mask = masks > 0.5
        if valid_mask.sum().item() == 0:
            continue

        with torch.no_grad():
            token_s1, token_s2 = tokenizer.encode(x, half=True)

        logits = model(token_s1, token_s2, x_stamp)  # (B,)

        loss_raw = criterion(logits, labels)  # (B,)
        loss_masked = loss_raw * masks
        loss = loss_masked.sum() / masks.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * valid_mask.sum().item()
        total_valid += valid_mask.sum().item()

        if log_interval > 0 and step % log_interval == 0:
            avg_so_far = total_loss / max(total_valid, 1)
            print(
                f"[Train] Step {step}/{num_batches} | "
                f"Avg Loss: {avg_so_far:.4f} | Valid={total_valid}",
                flush=True,
            )

    avg_loss = total_loss / max(total_valid, 1)
    return avg_loss


def evaluate(model, tokenizer, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_valid = 0
    total_correct = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_stamp = batch["x_stamp"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            valid_mask = masks > 0.5
            if valid_mask.sum().item() == 0:
                continue

            token_s1, token_s2 = tokenizer.encode(x, half=True)
            logits = model(token_s1, token_s2, x_stamp)

            loss_raw = criterion(logits, labels)
            loss_masked = loss_raw * masks
            loss = loss_masked.sum() / masks.sum().clamp(min=1.0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct = ((preds == labels) & valid_mask).sum().item()
            total = valid_mask.sum().item()

            total_loss += loss.item() * total
            total_correct += correct
            total_valid += total

    avg_loss = total_loss / max(total_valid, 1)
    da = total_correct / max(total_valid, 1)
    return avg_loss, da, total_valid


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward yearly finetuning for daily direction prediction."
    )

    # Data & time ranges
    parser.add_argument(
        "--full-data-path",
        type=str,
        required=True,
        help="Path to full BTCUSDT 5m CSV (2018-2025).",
    )
    parser.add_argument("--train-start-year", type=int, default=2018)
    parser.add_argument("--train-end-year", type=int, required=True)
    parser.add_argument("--test-year", type=int, required=True)

    # Horizon / context
    parser.add_argument("--lookback-window", type=int, default=400)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=288,
        help="Default=288 for 1-day horizon on 5-minute bars.",
    )
    parser.add_argument("--clip", type=float, default=5.0)

    # Volatility masking
    parser.add_argument(
        "--vol-window",
        type=int,
        default=288,
        help="Rolling window in bars for horizon-return volatility.",
    )
    parser.add_argument(
        "--vol-k",
        type=float,
        default=0.25,
        help="eps_t = max(vol_k * sigma_t, min_eps).",
    )
    parser.add_argument("--min-eps", type=float, default=0.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    # Model
    parser.add_argument(
        "--pretrained-tokenizer-path",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
    )
    parser.add_argument(
        "--pretrained-predictor-path",
        type=str,
        default="NeoQuasar/Kronos-small",
    )

    # Misc
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-dir", type=str, default="direction_daily_walkforward_ckpts")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Print training progress every N batches.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.full_data_path)
    required_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    if "timestamps" not in df.columns:
        raise ValueError("CSV must contain a 'timestamps' column.")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["year"] = df["timestamps"].dt.year

    train_mask = (df["year"] >= args.train_start_year) & (df["year"] <= args.train_end_year)
    test_mask = df["year"] == args.test_year

    df_trainval = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    if len(df_trainval) == 0:
        raise ValueError("No rows found for train+val years.")
    if len(df_test) == 0:
        raise ValueError("No rows found for test year.")

    tmp_dir = tempfile.mkdtemp(prefix="kronos_walkforward_")
    trainval_path = os.path.join(
        tmp_dir, f"btc_train_{args.train_start_year}_{args.train_end_year}.csv"
    )
    test_path = os.path.join(tmp_dir, f"btc_test_{args.test_year}.csv")

    df_trainval.to_csv(trainval_path, index=False)
    df_test.to_csv(test_path, index=False)

    test_ratio = max(0.0, 1.0 - args.train_ratio - args.val_ratio)

    train_dataset = DirectionKlineDataset(
        data_path=trainval_path,
        data_type="train",
        lookback_window=args.lookback_window,
        horizon_steps=args.horizon_steps,
        clip=args.clip,
        vol_window=args.vol_window,
        vol_k=args.vol_k,
        min_eps=args.min_eps,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )

    val_dataset = DirectionKlineDataset(
        data_path=trainval_path,
        data_type="val",
        lookback_window=args.lookback_window,
        horizon_steps=args.horizon_steps,
        clip=args.clip,
        vol_window=args.vol_window,
        vol_k=args.vol_k,
        min_eps=args.min_eps,
        seed=args.seed + 1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )

    test_dataset = DirectionKlineDataset(
        data_path=test_path,
        data_type="test",
        lookback_window=args.lookback_window,
        horizon_steps=args.horizon_steps,
        clip=args.clip,
        vol_window=args.vol_window,
        vol_k=args.vol_k,
        min_eps=args.min_eps,
        seed=args.seed + 2,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer_path)
    tokenizer.to(device)
    tokenizer.eval()

    model = build_kronos_direction_model(args.pretrained_predictor_path)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            criterion,
            device,
            log_interval=args.log_interval,
        )
        val_loss, val_da, val_count = evaluate(
            model, tokenizer, val_loader, criterion, device
        )

        print(
            f"[Year {args.train_start_year}-{args.train_end_year} -> Test {args.test_year}] "
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val DA: {val_da:.4f} (N={val_count})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                args.save_dir,
                f"best_daily_{args.train_start_year}_{args.train_end_year}_to_{args.test_year}.pt",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_path": args.pretrained_tokenizer_path,
                    "predictor_path": args.pretrained_predictor_path,
                    "train_start_year": args.train_start_year,
                    "train_end_year": args.train_end_year,
                    "test_year": args.test_year,
                    "horizon_steps": args.horizon_steps,
                },
                ckpt_path,
            )

    test_loss, test_da, test_count = evaluate(
        model, tokenizer, test_loader, criterion, device
    )
    print(
        f"[FINAL TEST] Train {args.train_start_year}-{args.train_end_year} -> "
        f"Test {args.test_year} | "
        f"Test Loss: {test_loss:.4f} | Test DA (MDA): {test_da:.4f} (N={test_count})"
    )


if __name__ == "__main__":
    main()
