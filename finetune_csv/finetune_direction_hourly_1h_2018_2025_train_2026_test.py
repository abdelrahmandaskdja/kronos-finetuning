import os
import argparse
import tempfile
import random
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import KronosTokenizer
from finetune_csv.direction_dataset import DirectionKlineDataset
from finetune_csv.kronos_direction_regression_model import (
    build_kronos_direction_regression_model,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_balanced_pos_weight_from_dataset(dataset: DirectionKlineDataset) -> float:
    """Estimate neg/pos ratio on valid (masked) train samples for BCE pos_weight."""
    if dataset.n_samples <= 0:
        return 1.0

    anchor = np.arange(dataset.n_samples, dtype=np.int64) + dataset.lookback_window - 1
    ret = dataset.horizon_ret[anchor]
    vol = dataset.horizon_vol[anchor]
    eps = np.maximum(dataset.vol_k * vol, dataset.min_eps)
    valid = np.isfinite(ret) & np.isfinite(vol) & (vol > 0.0) & (np.abs(ret) > eps)
    n_valid = int(valid.sum())
    if n_valid <= 0:
        return 1.0

    pos = int((ret[valid] > 0.0).sum())
    neg = n_valid - pos
    if pos <= 0 or neg <= 0:
        return 1.0
    return float(neg / pos)


def main():
    parser = argparse.ArgumentParser(
        description="1-hour BTCUSDT direction fine-tuning on Kronos: train 2018-2025, test 2026."
    )

    # Data & years
    parser.add_argument(
        "--full-data-path",
        type=str,
        required=True,
        help=(
            "CSV with BTCUSDT 1h candles, 2018-2026. Must have columns: "
            "timestamps, open, high, low, close, volume, amount."
        ),
    )
    parser.add_argument(
        "--train-start-year",
        type=int,
        default=2018,
        help="First training year (inclusive). Default: 2018.",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2025,
        help="Last training year (inclusive). Default: 2025.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2026,
        help="Test year (exclusive to training). Default: 2026.",
    )

    # Horizon / context (1-hour horizon = 1 bar ahead)
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=200,
        help="Number of past 1h bars as context.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=1,
        help="Bars ahead for target. For 1-hour horizon on 1h data: 1.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Clipping value after feature normalization.",
    )

    # Volatility-based masking (on 1h returns)
    parser.add_argument(
        "--vol-window",
        type=int,
        default=168,
        help="Rolling window (in 1h bars) for volatility of horizon returns (e.g., 168 = 1 week).",
    )
    parser.add_argument(
        "--vol-k",
        type=float,
        default=0.25,
        help="Scale factor k: eps_t = max(vol_k * sigma_t, min_eps).",
    )
    parser.add_argument(
        "--min-eps",
        type=float,
        default=0.0,
        help="Minimum epsilon floor for eps_t.",
    )

    # Training / validation split within 2018-2025 (time-based split)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of 2018-2025 data used for train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of 2018-2025 data used for validation.",
    )

    # Optimization
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional positive class weight for BCE. If None, no weighting.",
    )
    parser.add_argument(
        "--w-mse",
        type=float,
        default=1.0,
        help="Weight for MSE loss on horizon return.",
    )
    parser.add_argument(
        "--w-dir",
        type=float,
        default=1.0,
        help="Weight for BCE loss on direction.",
    )
    parser.add_argument(
        "--optimize-mda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable aggressive MDA optimization (adaptive direction-first loss schedule).",
    )
    parser.add_argument(
        "--auto-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-compute BCE pos_weight from train split if --pos-weight is not provided.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Focal-BCE gamma for direction loss. 0 disables focal reweighting.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for BCE targets in [0, 1).",
    )
    parser.add_argument(
        "--mda-w-dir-floor",
        type=float,
        default=2.0,
        help="Minimum direction-loss weight when --optimize-mda is enabled.",
    )
    parser.add_argument(
        "--mda-w-mse-cap",
        type=float,
        default=0.2,
        help="Maximum MSE-loss weight when --optimize-mda is enabled.",
    )
    parser.add_argument(
        "--mda-target-train",
        type=float,
        default=0.58,
        help="Target training DA@0.5 used to decide if direction pressure should increase.",
    )
    parser.add_argument(
        "--mda-dir-boost",
        type=float,
        default=1.25,
        help="Multiplier for direction-loss weight when validation MDA stalls.",
    )
    parser.add_argument(
        "--mda-dir-max",
        type=float,
        default=12.0,
        help="Max direction-loss weight in adaptive MDA mode.",
    )
    parser.add_argument(
        "--mda-mse-decay",
        type=float,
        default=0.7,
        help="Multiplier for MSE-loss weight when validation MDA stalls.",
    )
    parser.add_argument(
        "--mda-mse-min",
        type=float,
        default=0.0,
        help="Minimum MSE-loss weight in adaptive MDA mode.",
    )
    parser.add_argument(
        "--mda-plateau-patience",
        type=int,
        default=2,
        help="Consecutive non-improving epochs before adaptive MDA reweighting kicks in.",
    )
    parser.add_argument(
        "--mda-early-stop-patience",
        type=int,
        default=8,
        help="Stop training after this many non-improving epochs on val best DA.",
    )
    parser.add_argument(
        "--mda-min-delta",
        type=float,
        default=5e-4,
        help="Minimum val-best-DA improvement treated as progress.",
    )
    parser.add_argument(
        "--lr-plateau",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReduceLROnPlateau scheduler on val best DA.",
    )
    parser.add_argument(
        "--lr-drop-factor",
        type=float,
        default=0.6,
        help="LR multiplier when plateau scheduler triggers.",
    )
    parser.add_argument(
        "--lr-drop-patience",
        type=int,
        default=2,
        help="Plateau epochs before reducing LR.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
        help="Lower bound for LR scheduler.",
    )

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
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=512,
        help="Hidden size of direction head MLP. Use <=0 for single linear head.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
        help="Dropout in direction head MLP.",
    )
    parser.add_argument(
        "--head-use-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply LayerNorm before direction head.",
    )
    parser.add_argument(
        "--head-pooling",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="How to pool Kronos sequence features before head.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze Kronos backbone and train only direction head.",
    )

    # Threshold search
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.3,
        help="Min probability threshold to scan.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=0.7,
        help="Max probability threshold to scan.",
    )
    parser.add_argument(
        "--tau-steps",
        type=int,
        default=101,
        help="Number of steps between tau-min and tau-max.",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string, e.g. 'cuda' or 'cpu'. Empty = auto.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="direction_hourly_1h_2018_2025_train_2026_test",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON path for final metrics.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume/warm-start training.",
    )
    parser.add_argument(
        "--resume-start-epoch",
        type=int,
        default=-1,
        help="Optional manual epoch to start from when resuming old checkpoints.",
    )

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio > 1.0 + 1e-6:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    if args.tau_steps < 2:
        raise ValueError("tau_steps must be >= 2")
    if args.tau_min > args.tau_max:
        raise ValueError("tau_min must be <= tau_max")
    if args.focal_gamma < 0.0:
        raise ValueError("focal-gamma must be >= 0.")
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("label-smoothing must be in [0, 1).")
    if args.mda_dir_boost < 1.0:
        raise ValueError("mda-dir-boost must be >= 1.0.")
    if args.mda_dir_max <= 0.0:
        raise ValueError("mda-dir-max must be > 0.")
    if not (0.0 < args.mda_mse_decay <= 1.0):
        raise ValueError("mda-mse-decay must be in (0, 1].")
    if args.mda_plateau_patience < 1:
        raise ValueError("mda-plateau-patience must be >= 1.")
    if args.mda_early_stop_patience < 0:
        raise ValueError("mda-early-stop-patience must be >= 0.")
    if args.lr_drop_patience < 1:
        raise ValueError("lr-drop-patience must be >= 1.")

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------
    # Load full 1h data and split by year
    # ----------------------------------------
    df = pd.read_csv(args.full_data_path)
    required_cols = {"timestamps", "open", "high", "low", "close", "volume", "amount"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

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

    print(f"Train/Val rows: {len(df_trainval)}, Test rows: {len(df_test)}")

    tmp_dir = tempfile.mkdtemp(prefix="kronos_hourly_1h_2018_2026_")
    trainval_path = os.path.join(
        tmp_dir,
        f"btcusdt_1h_trainval_{args.train_start_year}_{args.train_end_year}.csv",
    )
    test_path = os.path.join(tmp_dir, f"btcusdt_1h_test_{args.test_year}.csv")

    df_trainval.to_csv(trainval_path, index=False)
    df_test.to_csv(test_path, index=False)

    # ----------------------------------------
    # Datasets and loaders
    # ----------------------------------------
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

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    # ----------------------------------------
    # Model, tokenizer, optimizer, loss
    # ----------------------------------------
    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer_path)
    tokenizer.to(device)
    tokenizer.eval()

    model = build_kronos_direction_regression_model(
        pretrained_predictor_path=args.pretrained_predictor_path,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_use_layernorm=args.head_use_layernorm,
        pooling=args.head_pooling,
    )
    model.to(device)
    if args.freeze_backbone:
        for p in model.kronos.parameters():
            p.requires_grad_(False)

    current_w_dir = float(args.w_dir)
    current_w_mse = float(args.w_mse)
    if args.optimize_mda:
        current_w_dir = max(current_w_dir, float(args.mda_w_dir_floor))
        current_w_mse = min(current_w_mse, float(args.mda_w_mse_cap))

    effective_pos_weight = args.pos_weight
    if effective_pos_weight is None and args.auto_pos_weight:
        inferred_pos_weight = compute_balanced_pos_weight_from_dataset(train_dataset)
        effective_pos_weight = float(np.clip(inferred_pos_weight, 0.25, 16.0))
        print(
            f"Auto pos_weight from train split: {effective_pos_weight:.4f} "
            "(clipped to [0.25, 16.0])"
        )

    if effective_pos_weight is not None:
        pos_w = torch.tensor(float(effective_pos_weight), device=device, dtype=torch.float32)
        dir_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
    else:
        dir_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def direction_loss_raw(logits, labels):
        labels_for_loss = labels
        if args.label_smoothing > 0.0:
            labels_for_loss = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

        raw = dir_criterion(logits, labels_for_loss)
        if args.focal_gamma > 0.0:
            probs = torch.sigmoid(logits)
            pt = torch.where(labels > 0.5, probs, 1.0 - probs)
            focal = torch.pow((1.0 - pt).clamp(min=1e-6), args.focal_gamma)
            raw = raw * focal
        return raw

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(
        "Model params | "
        f"total={total_param_count:,} trainable={trainable_param_count:,} "
        f"freeze_backbone={args.freeze_backbone} "
        f"head_hidden_dim={args.head_hidden_dim} "
        f"head_dropout={args.head_dropout} "
        f"head_use_layernorm={args.head_use_layernorm} "
        f"head_pooling={args.head_pooling} "
        f"optimize_mda={args.optimize_mda} "
        f"w_dir_start={current_w_dir:.4f} w_mse_start={current_w_mse:.4f} "
        f"focal_gamma={args.focal_gamma:.3f} label_smoothing={args.label_smoothing:.3f}"
    )
    if trainable_param_count == 0:
        raise RuntimeError("No trainable parameters. Disable --freeze-backbone.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = None
    if args.lr_plateau:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_drop_factor,
            patience=args.lr_drop_patience,
            min_lr=args.min_learning_rate,
        )

    # ----------------------------------------
    # Training / validation helpers
    # ----------------------------------------
    def train_one_epoch(cur_w_mse: float, cur_w_dir: float):
        model.train()
        total_loss = 0.0
        total_valid = 0
        total_correct = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            x_stamp = batch["x_stamp"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            rets = batch["ret"].to(device)

            valid_mask = masks > 0.5
            if valid_mask.sum().item() == 0:
                continue

            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(x, half=True)

            ret_pred, dir_logits = model(token_s1, token_s2, x_stamp)

            dir_raw = direction_loss_raw(dir_logits, labels)
            mse_raw = F.mse_loss(ret_pred, rets, reduction="none")
            denom = masks.sum().clamp(min=1.0)
            dir_loss = (dir_raw * masks).sum() / denom
            mse_loss = (mse_raw * masks).sum() / denom
            loss = cur_w_mse * mse_loss + cur_w_dir * dir_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            probs = torch.sigmoid(dir_logits)
            preds = (probs >= 0.5).float()
            total_correct += ((preds == labels) * valid_mask).sum().item()
            total_loss += loss.item() * valid_mask.sum().item()
            total_valid += valid_mask.sum().item()

        avg_loss = total_loss / max(total_valid, 1)
        train_da_05 = total_correct / max(total_valid, 1)
        return avg_loss, train_da_05

    def evaluate_and_search_tau(loader, tau_min, tau_max, tau_steps, cur_w_mse, cur_w_dir):
        model.eval()
        total_loss = 0.0
        total_valid = 0

        all_probs = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                x_stamp = batch["x_stamp"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)
                rets = batch["ret"].to(device)

                valid_mask = masks > 0.5
                if valid_mask.sum().item() == 0:
                    continue

                token_s1, token_s2 = tokenizer.encode(x, half=True)
                ret_pred, dir_logits = model(token_s1, token_s2, x_stamp)

                dir_raw = direction_loss_raw(dir_logits, labels)
                mse_raw = F.mse_loss(ret_pred, rets, reduction="none")
                denom = masks.sum().clamp(min=1.0)
                dir_loss = (dir_raw * masks).sum() / denom
                mse_loss = (mse_raw * masks).sum() / denom
                loss = cur_w_mse * mse_loss + cur_w_dir * dir_loss

                total_loss += loss.item() * valid_mask.sum().item()
                total_valid += valid_mask.sum().item()

                probs = torch.sigmoid(dir_logits)
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_masks.append(masks.detach().cpu())

        if total_valid == 0:
            return 0.0, 0.0, 0.5, 0.0, 0

        avg_loss = total_loss / total_valid

        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)
        masks = torch.cat(all_masks)
        valid = masks > 0.5

        probs = probs[valid]
        labels = labels[valid]
        n_valid = labels.shape[0]

        preds_05 = (probs >= 0.5).float()
        da_05 = (preds_05 == labels).float().mean().item()

        tau_values = np.linspace(tau_min, tau_max, tau_steps)
        best_tau = 0.5
        best_da = da_05

        probs_np = probs.numpy()
        labels_np = labels.numpy()

        for tau in tau_values:
            preds = (probs_np >= tau).astype(np.float32)
            da = (preds == labels_np).mean()
            if da > best_da:
                best_da = float(da)
                best_tau = float(tau)

        return avg_loss, da_05, best_tau, best_da, int(n_valid)

    def evaluate_with_tau(loader, tau, cur_w_mse, cur_w_dir):
        model.eval()
        total_loss = 0.0
        total_valid = 0
        total_correct = 0
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                x_stamp = batch["x_stamp"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)
                rets = batch["ret"].to(device)

                valid_mask = masks > 0.5
                if valid_mask.sum().item() == 0:
                    continue

                token_s1, token_s2 = tokenizer.encode(x, half=True)
                ret_pred, dir_logits = model(token_s1, token_s2, x_stamp)

                dir_raw = direction_loss_raw(dir_logits, labels)
                mse_raw = F.mse_loss(ret_pred, rets, reduction="none")
                denom = masks.sum().clamp(min=1.0)
                dir_loss = (dir_raw * masks).sum() / denom
                mse_loss = (mse_raw * masks).sum() / denom
                loss = cur_w_mse * mse_loss + cur_w_dir * dir_loss

                probs = torch.sigmoid(dir_logits)
                preds = (probs >= tau).float()

                correct = ((preds == labels) * valid_mask).sum().item()
                total = valid_mask.sum().item()

                total_loss += loss.item() * total
                total_correct += correct
                total_valid += total
                all_labels.append(labels[valid_mask].detach().cpu())

        if total_valid == 0:
            return 0.0, 0.0, 0.0, 0

        avg_loss = total_loss / total_valid
        da = total_correct / total_valid

        labels_cat = torch.cat(all_labels) if all_labels else torch.zeros(0)
        if labels_cat.numel() > 0:
            frac_up = labels_cat.mean().item()
            baseline_da = max(frac_up, 1.0 - frac_up)
        else:
            baseline_da = 0.0

        return avg_loss, da, baseline_da, int(total_valid)

    # ----------------------------------------
    # Training loop (optimize val DA)
    # ----------------------------------------
    start_epoch = 0
    best_val_da = -1.0
    best_val_tau = 0.5
    best_ckpt_path = None
    best_epoch = -1
    epochs_no_improve = 0
    plateau_epochs = 0

    os.makedirs(args.save_dir, exist_ok=True)
    default_ckpt_path = os.path.join(
        args.save_dir,
        f"best_hourly_1h_train_{args.train_start_year}_{args.train_end_year}_test_{args.test_year}.pt",
    )

    resume_checkpoint = args.resume_checkpoint.strip()
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"--resume-checkpoint not found: {resume_checkpoint}")
        resume_obj = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(resume_obj["model_state_dict"])
        if "optimizer_state_dict" in resume_obj:
            try:
                optimizer.load_state_dict(resume_obj["optimizer_state_dict"])
            except Exception as exc:
                print(f"Warning: optimizer state load failed, continuing with fresh optimizer. {exc}")
        start_epoch = int(resume_obj.get("epoch", -1)) + 1
        if args.resume_start_epoch >= 0:
            start_epoch = int(args.resume_start_epoch)
        best_val_da = float(resume_obj.get("best_val_da", -1.0))
        best_val_tau = float(resume_obj.get("best_val_tau", 0.5))
        current_w_dir = float(resume_obj.get("current_w_dir", current_w_dir))
        current_w_mse = float(resume_obj.get("current_w_mse", current_w_mse))
        best_ckpt_path = resume_checkpoint
        print(
            f"Resumed from checkpoint: {resume_checkpoint} | "
            f"start_epoch={start_epoch} | best_val_da={best_val_da:.4f} | best_val_tau={best_val_tau:.3f} "
            f"| w_dir={current_w_dir:.4f} | w_mse={current_w_mse:.4f}"
        )

    if best_ckpt_path is None:
        best_ckpt_path = default_ckpt_path

    if start_epoch >= args.num_epochs:
        print(
            f"Checkpoint epoch already >= --num-epochs ({start_epoch} >= {args.num_epochs}). "
            "Skipping training loop."
        )

    for epoch in range(start_epoch, args.num_epochs):
        train_dataset.set_epoch_seed(epoch)

        train_loss, train_da_05 = train_one_epoch(current_w_mse, current_w_dir)
        val_loss, val_da_05, val_best_tau, val_best_da, val_n = evaluate_and_search_tau(
            val_loader,
            args.tau_min,
            args.tau_max,
            args.tau_steps,
            current_w_mse,
            current_w_dir,
        )

        if lr_scheduler is not None:
            lr_scheduler.step(val_best_da)
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train DA@0.5: {train_da_05:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val DA@0.5: {val_da_05:.4f} | "
            f"Val Best Tau: {val_best_tau:.3f} | "
            f"Val Best DA: {val_best_da:.4f} (N={val_n}) | "
            f"w_dir={current_w_dir:.4f} | w_mse={current_w_mse:.4f} | lr={current_lr:.2e}"
        )

        improved = val_best_da > (best_val_da + args.mda_min_delta)
        if improved:
            best_val_da = val_best_da
            best_val_tau = val_best_tau
            best_ckpt_path = default_ckpt_path
            best_epoch = int(epoch)
            epochs_no_improve = 0
            plateau_epochs = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tokenizer_path": args.pretrained_tokenizer_path,
                    "predictor_path": args.pretrained_predictor_path,
                    "train_start_year": args.train_start_year,
                    "train_end_year": args.train_end_year,
                    "test_year": args.test_year,
                    "horizon_steps": args.horizon_steps,
                    "best_val_tau": best_val_tau,
                    "best_val_da": best_val_da,
                    "seed": args.seed,
                    "head_hidden_dim": args.head_hidden_dim,
                    "head_dropout": args.head_dropout,
                    "head_use_layernorm": args.head_use_layernorm,
                    "head_pooling": args.head_pooling,
                    "w_mse": args.w_mse,
                    "w_dir": args.w_dir,
                    "current_w_mse": current_w_mse,
                    "current_w_dir": current_w_dir,
                    "optimize_mda": args.optimize_mda,
                    "pos_weight": effective_pos_weight,
                    "focal_gamma": args.focal_gamma,
                    "label_smoothing": args.label_smoothing,
                    "best_epoch": best_epoch,
                },
                best_ckpt_path,
            )
            print(f"  -> New best checkpoint saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            plateau_epochs += 1

            if args.optimize_mda and plateau_epochs >= args.mda_plateau_patience:
                if train_da_05 < args.mda_target_train:
                    new_w_dir = min(current_w_dir * args.mda_dir_boost, args.mda_dir_max)
                    new_w_mse = max(current_w_mse * args.mda_mse_decay, args.mda_mse_min)
                    if (new_w_dir != current_w_dir) or (new_w_mse != current_w_mse):
                        print(
                            "  -> MDA plateau detected; increasing direction pressure: "
                            f"w_dir {current_w_dir:.4f}->{new_w_dir:.4f}, "
                            f"w_mse {current_w_mse:.4f}->{new_w_mse:.4f}"
                        )
                    current_w_dir = new_w_dir
                    current_w_mse = new_w_mse
                plateau_epochs = 0

            if args.optimize_mda and args.mda_early_stop_patience > 0:
                if epochs_no_improve >= args.mda_early_stop_patience:
                    print(
                        f"Early stop: no val-best-DA improvement for {epochs_no_improve} epochs "
                        f"(patience={args.mda_early_stop_patience})."
                    )
                    break

    print(f"\nBest validation DA: {best_val_da:.4f} at tau={best_val_tau:.3f}")
    print("Evaluating on TEST year...")

    if best_ckpt_path is not None:
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_tau = ckpt.get("best_val_tau", best_val_tau)
        current_w_mse = float(ckpt.get("current_w_mse", current_w_mse))
        current_w_dir = float(ckpt.get("current_w_dir", current_w_dir))

    test_loss, test_da, test_baseline_da, test_n = evaluate_with_tau(
        test_loader,
        tau=best_val_tau,
        cur_w_mse=current_w_mse,
        cur_w_dir=current_w_dir,
    )

    print(
        f"[FINAL TEST] Train {args.train_start_year}-{args.train_end_year} -> "
        f"Test {args.test_year} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test DA (MDA) at tau={best_val_tau:.3f}: {test_da:.4f} "
        f"(baseline={test_baseline_da:.4f}, N={test_n})"
    )

    metrics = {
        "full_data_path": args.full_data_path,
        "train_start_year": int(args.train_start_year),
        "train_end_year": int(args.train_end_year),
        "test_year": int(args.test_year),
        "lookback_window": int(args.lookback_window),
        "horizon_steps": int(args.horizon_steps),
        "vol_window": int(args.vol_window),
        "vol_k": float(args.vol_k),
        "min_eps": float(args.min_eps),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "pos_weight": None if effective_pos_weight is None else float(effective_pos_weight),
        "w_mse": float(args.w_mse),
        "w_dir": float(args.w_dir),
        "effective_w_mse": float(current_w_mse),
        "effective_w_dir": float(current_w_dir),
        "optimize_mda": bool(args.optimize_mda),
        "auto_pos_weight": bool(args.auto_pos_weight),
        "focal_gamma": float(args.focal_gamma),
        "label_smoothing": float(args.label_smoothing),
        "mda_target_train": float(args.mda_target_train),
        "mda_dir_boost": float(args.mda_dir_boost),
        "mda_dir_max": float(args.mda_dir_max),
        "mda_mse_decay": float(args.mda_mse_decay),
        "mda_mse_min": float(args.mda_mse_min),
        "mda_plateau_patience": int(args.mda_plateau_patience),
        "mda_early_stop_patience": int(args.mda_early_stop_patience),
        "mda_min_delta": float(args.mda_min_delta),
        "best_epoch": int(best_epoch),
        "freeze_backbone": bool(args.freeze_backbone),
        "head_hidden_dim": int(args.head_hidden_dim),
        "head_dropout": float(args.head_dropout),
        "head_use_layernorm": bool(args.head_use_layernorm),
        "head_pooling": str(args.head_pooling),
        "pretrained_tokenizer_path": str(args.pretrained_tokenizer_path),
        "pretrained_predictor_path": str(args.pretrained_predictor_path),
        "tau_min": float(args.tau_min),
        "tau_max": float(args.tau_max),
        "tau_steps": int(args.tau_steps),
        "seed": int(args.seed),
        "device": str(device),
        "best_checkpoint_path": best_ckpt_path,
        "best_val_da": float(best_val_da),
        "best_val_tau": float(best_val_tau),
        "test_loss": float(test_loss),
        "test_mda": float(test_da),
        "test_majority_baseline": float(test_baseline_da),
        "test_valid_masked_samples": int(test_n),
    }

    output_json = args.output_json.strip()
    if not output_json:
        output_json = os.path.join(args.save_dir, "metrics.json")
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
