# Fine-tunes a small direction head (optionally with frozen Kronos backbone)
# for a single horizon (--horizon-steps), using BCE on log-return direction.
# By default, the dataset computes local eps_t = max(vol_k * sigma_t, min_eps), where
# sigma_t is rolling std of horizon returns over --vol-window bars, and samples with
# |ret| <= eps_t are masked out (ignored). Pass --no-mask-ambiguous to score all rows.
import os
import argparse
import random
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import KronosTokenizer
from finetune_csv.direction_dataset import DirectionKlineDataset
from finetune_csv.kronos_direction_model import build_kronos_direction_model
from finetune_csv.training_charts import save_training_dashboard


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEma:
    """CPU-backed EMA of model weights for cheap evaluation-time smoothing."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(np.clip(decay, 0.0, 0.999999))
        self.shadow_state = None
        self.backup_state = None
        if self.decay > 0.0:
            self.shadow_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    def has_state(self) -> bool:
        return self.shadow_state is not None

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.shadow_state is None:
            return
        for name, tensor in model.state_dict().items():
            shadow_tensor = self.shadow_state[name]
            current_cpu = tensor.detach().cpu()
            if torch.is_floating_point(current_cpu):
                shadow_tensor.mul_(self.decay).add_(current_cpu, alpha=1.0 - self.decay)
            else:
                shadow_tensor.copy_(current_cpu)

    @torch.no_grad()
    def store(self, model: nn.Module):
        self.backup_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        if self.shadow_state is None:
            return
        target_state = model.state_dict()
        for name, tensor in target_state.items():
            tensor.copy_(self.shadow_state[name].to(device=tensor.device, dtype=tensor.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup_state is None:
            return
        target_state = model.state_dict()
        for name, tensor in target_state.items():
            tensor.copy_(self.backup_state[name].to(device=tensor.device, dtype=tensor.dtype))
        self.backup_state = None


def train_one_epoch(
    model,
    tokenizer,
    train_loader,
    optimizer,
    criterion,
    device,
    global_step_start=0,
    step_rows=None,
    log_interval=100,
    scheduler=None,
    scheduler_step_mode="none",
    ema_helper=None,
    ema_start_step=0,
):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_valid = 0
    global_step = int(global_step_start)
    log_every = max(int(log_interval), 1)
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader, start=1):
        x = batch["x"].to(device)  # (B, L, 6)
        x_stamp = batch["x_stamp"].to(device)  # (B, L, 5)
        labels = batch["label"].to(device)  # (B,)
        masks = batch["mask"].to(device)  # (B,)

        scored_mask = masks > 0.5
        if scored_mask.sum().item() == 0:
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
        global_step += 1

        if scheduler is not None and scheduler_step_mode == "batch":
            scheduler.step()

        if ema_helper is not None and global_step >= int(ema_start_step):
            ema_helper.update(model)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        batch_valid = int(scored_mask.sum().item())
        batch_correct = int(((preds == labels) & scored_mask).sum().item())

        running_loss += loss.item() * batch_valid
        total_correct += batch_correct
        total_valid += batch_valid

        if batch_idx % log_every == 0 or batch_idx == num_batches:
            running_avg_loss = running_loss / max(total_valid, 1)
            running_avg_acc = total_correct / max(total_valid, 1)
            batch_acc = batch_correct / max(batch_valid, 1)
            print(
                f"[Train] Batch {batch_idx}/{num_batches} | "
                f"Global Step {global_step} | "
                f"Batch Loss {loss.item():.4f} | "
                f"Batch Acc {batch_acc:.4f} | "
                f"Running Loss {running_avg_loss:.4f} | "
                f"Running Acc {running_avg_acc:.4f} | "
                f"Scored Rows {batch_valid}",
                flush=True,
            )

        if step_rows is not None and (batch_idx % log_every == 0 or batch_idx == num_batches):
            step_rows.append(
                {
                    "global_step": float(global_step),
                    "train_loss": float(loss.item()),
                    "train_direction_accuracy_pct": float(100.0 * batch_correct / max(batch_valid, 1)),
                    "valid_rows": float(batch_valid),
                }
            )

    train_loss = running_loss / max(total_valid, 1)
    train_da = total_correct / max(total_valid, 1)
    return train_loss, train_da, global_step


def evaluate(model, tokenizer, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)  # (B, L, 6)
            x_stamp = batch["x_stamp"].to(device)  # (B, L, 5)
            labels = batch["label"].to(device)  # (B,)
            masks = batch["mask"].to(device)  # (B,)

            valid = masks > 0.5
            if valid.sum().item() == 0:
                continue

            token_s1, token_s2 = tokenizer.encode(x, half=True)
            logits = model(token_s1, token_s2, x_stamp)  # (B,)

            loss_raw = criterion(logits, labels)  # (B,)
            loss_masked = loss_raw * masks
            loss = loss_masked.sum() / masks.sum().clamp(min=1.0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct = ((preds == labels) & valid).sum().item()
            total = valid.sum().item()

            total_loss += loss.item() * total
            total_correct += correct
            total_valid += total

    avg_val_loss = total_loss / max(total_valid, 1)
    da = total_correct / max(total_valid, 1)
    return avg_val_loss, da


def _metric_improved(current, best, *, higher_is_better: bool, min_delta: float = 0.0) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if higher_is_better:
        return float(current) > float(best) + float(min_delta)
    return float(current) < float(best) - float(min_delta)


def configure_trainable_parameters(model, freeze_backbone: bool, unfreeze_last_n: int):
    """Configure backbone freezing / partial unfreezing policy.

    Policy:
    - Direction head is always trainable.
    - If freeze_backbone=True, Kronos backbone is frozen.
    - If unfreeze_last_n>=0, Kronos is partially trainable: only last N transformer
      blocks plus {norm, dep_layer, output_adapter(if present)} are trainable.
      This policy overrides the blanket freeze/all-train behavior for Kronos.
    """
    for p in model.parameters():
        p.requires_grad_(True)

    kronos = model.kronos
    n_total_blocks = len(getattr(kronos, "transformer", []))
    trainable_transformer = "all"

    if freeze_backbone:
        for p in kronos.parameters():
            p.requires_grad_(False)
        trainable_transformer = "none"

    if unfreeze_last_n >= 0:
        for p in kronos.parameters():
            p.requires_grad_(False)

        n_keep = min(max(unfreeze_last_n, 0), n_total_blocks)
        start_idx = n_total_blocks - n_keep

        if n_keep > 0:
            for block in kronos.transformer[start_idx:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            trainable_transformer = f"{start_idx}-{n_total_blocks - 1}"
        else:
            trainable_transformer = "none"

        for module_name in ("norm", "dep_layer", "output_adapter"):
            module = getattr(kronos, module_name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(True)

    for p in model.direction_head.parameters():
        p.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "freeze_backbone": bool(freeze_backbone),
        "unfreeze_last_n_transformer_layers": int(unfreeze_last_n),
        "kronos_transformer_total_blocks": int(n_total_blocks),
        "kronos_trainable_transformer_blocks": trainable_transformer,
    }


def _resolve_warmup_steps(total_optimizer_steps: int, warmup_ratio: float) -> int:
    total_optimizer_steps = max(int(total_optimizer_steps), 1)
    warmup_ratio = float(np.clip(warmup_ratio, 0.0, 0.95))
    warmup_steps = int(round(total_optimizer_steps * warmup_ratio))
    return max(0, min(warmup_steps, max(total_optimizer_steps - 1, 0)))


def _build_scheduler(optimizer, args, *, steps_per_epoch: int):
    scheduler_name = str(args.scheduler).strip().lower()
    if scheduler_name in {"", "none"}:
        return None, "none"

    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_epochs = max(int(args.num_epochs), 1)
    total_optimizer_steps = max(steps_per_epoch * total_epochs, 1)
    warmup_steps = _resolve_warmup_steps(total_optimizer_steps, args.scheduler_warmup_ratio)
    min_lr_ratio = float(np.clip(args.scheduler_min_lr_ratio, 0.0, 1.0))
    base_lr = float(args.learning_rate)

    if scheduler_name == "onecycle":
        pct_start = float(np.clip(max(args.scheduler_warmup_ratio, 0.03), 0.03, 0.9))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            pct_start=pct_start,
            div_factor=10.0,
            final_div_factor=max(1.0 / max(min_lr_ratio, 1e-4), 10.0),
        )
        return scheduler, "batch"

    if scheduler_name == "cosine":
        cosine_steps = max(total_optimizer_steps - warmup_steps, 1)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=base_lr * min_lr_ratio,
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler
        return scheduler, "batch"

    if scheduler_name == "plateau":
        plateau_metric = str(args.plateau_metric).strip().lower()
        mode = "max" if plateau_metric == "val_direction_accuracy" else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(np.clip(args.plateau_factor, 1e-4, 0.9999)),
            patience=max(int(args.plateau_patience), 0),
            min_lr=base_lr * min_lr_ratio,
        )
        return scheduler, "epoch"

    raise ValueError(
        f"Unsupported --scheduler={args.scheduler}. Use one of: none, cosine, onecycle, plateau."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune KronosDirectionModel on single-asset OHLCV CSV."
    )
    parser.add_argument("--data-path", type=str, required=True, help="CSV of 5m OHLCV")
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
        help="Direction head hidden size. Use <=0 for linear head only.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
        help="Dropout probability in direction head.",
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
        help="Sequence pooling for Kronos context before direction head.",
    )
    parser.add_argument("--lookback-window", type=int, default=400)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument(
        "--vol-window",
        type=int,
        default=288,
        help="Rolling window (in bars) for horizon return volatility.",
    )
    parser.add_argument(
        "--vol-k",
        type=float,
        default=0.25,
        help="Scale factor k: eps_t = vol_k * rolling_std.",
    )
    parser.add_argument(
        "--min-eps",
        type=float,
        default=0.0,
        help="Minimum epsilon floor for eps_t.",
    )
    parser.add_argument(
        "--mask-ambiguous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore ambiguous low-magnitude moves. Disable to train on every finite return.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="direction_checkpoints")
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Kronos backbone parameters.",
    )
    parser.add_argument(
        "--unfreeze-last-n-transformer-layers",
        type=int,
        default=-1,
        help=(
            "If >=0, only last N Kronos transformer blocks are trainable "
            "(plus norm/dep_layer/output_adapter if present). "
            "Set 0 for head-only training."
        ),
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["mda_cross_entropy", "bce"],
        default="mda_cross_entropy",
        help="Directional classification loss. 'mda_cross_entropy' maps to BCEWithLogits.",
    )
    parser.add_argument(
        "--save-metrics-json",
        type=str,
        default="",
        help="Optional output JSON for epoch-wise training/validation metrics.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Record direction-head step metrics every N training batches.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "onecycle", "plateau"],
        help="Learning-rate schedule for direction-head training.",
    )
    parser.add_argument(
        "--scheduler-warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup fraction for batch-wise schedulers.",
    )
    parser.add_argument(
        "--scheduler-min-lr-ratio",
        type=float,
        default=0.1,
        help="Minimum LR ratio relative to base LR for schedulers that decay.",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="LR decay factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="Patience for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_direction_accuracy"],
        help="Metric used by the plateau scheduler.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_direction_accuracy"],
        help="Metric used to select best_direction_model.pt.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without checkpoint-metric improvement. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum checkpoint-metric improvement required to reset patience.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for evaluation-time smoothing. 0 disables EMA.",
    )
    parser.add_argument(
        "--ema-start-step",
        type=int,
        default=0,
        help="Global optimizer step at which EMA updates begin.",
    )
    parser.add_argument(
        "--ema-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate checkpoint metrics using EMA weights when EMA is enabled.",
    )
    parser.add_argument(
        "--save-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the latest raw weights to last_direction_model.pt.",
    )
    parser.add_argument(
        "--save-best-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the highest train-accuracy checkpoint to best_train_direction_model.pt.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    split_sum = float(args.train_ratio) + float(args.val_ratio)
    test_ratio = 1.0 - split_sum
    if test_ratio < -1e-9:
        raise ValueError("train_ratio + val_ratio must be <= 1.0.")
    test_ratio = max(0.0, test_ratio)

    train_dataset = DirectionKlineDataset(
        data_path=args.data_path,
        data_type="train",
        lookback_window=args.lookback_window,
        horizon_steps=args.horizon_steps,
        clip=args.clip,
        vol_window=args.vol_window,
        vol_k=args.vol_k,
        min_eps=args.min_eps,
        mask_ambiguous=args.mask_ambiguous,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )
    val_dataset = DirectionKlineDataset(
        data_path=args.data_path,
        data_type="val",
        lookback_window=args.lookback_window,
        horizon_steps=args.horizon_steps,
        clip=args.clip,
        vol_window=args.vol_window,
        vol_k=args.vol_k,
        min_eps=args.min_eps,
        mask_ambiguous=args.mask_ambiguous,
        seed=args.seed + 1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
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

    tokenizer = KronosTokenizer.from_pretrained(args.pretrained_tokenizer_path)
    tokenizer.to(device)
    tokenizer.eval()

    model = build_kronos_direction_model(
        pretrained_predictor_path=args.pretrained_predictor_path,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_use_layernorm=args.head_use_layernorm,
        pooling=args.head_pooling,
    )
    model.to(device)

    param_setup = configure_trainable_parameters(
        model=model,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n_transformer_layers,
    )
    print(
        "Parameter setup | "
        f"total={param_setup['total_params']:,} | "
        f"trainable={param_setup['trainable_params']:,} | "
        f"frozen={param_setup['frozen_params']:,} | "
        f"freeze_backbone={param_setup['freeze_backbone']} | "
        f"unfreeze_last_n={param_setup['unfreeze_last_n_transformer_layers']} | "
        f"kronos_blocks={param_setup['kronos_transformer_total_blocks']} | "
        f"trainable_blocks={param_setup['kronos_trainable_transformer_blocks']}"
    )
    if param_setup["trainable_params"] == 0:
        raise RuntimeError("No trainable parameters found; cannot fine-tune.")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=float(args.weight_decay),
    )
    if args.loss_type in ("mda_cross_entropy", "bce"):
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss-type: {args.loss_type}")

    checkpoint_metric = str(args.checkpoint_metric).strip().lower()
    higher_is_better = checkpoint_metric == "val_direction_accuracy"
    scheduler, scheduler_step_mode = _build_scheduler(
        optimizer=optimizer,
        args=args,
        steps_per_epoch=len(train_loader),
    )
    ema_helper = ModelEma(model, args.ema_decay) if float(args.ema_decay) > 0.0 else None

    print(
        "Optimizer setup | "
        f"lr={args.learning_rate:.6g} | "
        f"weight_decay={args.weight_decay:.6g} | "
        f"scheduler={args.scheduler} | "
        f"scheduler_step_mode={scheduler_step_mode} | "
        f"checkpoint_metric={checkpoint_metric} | "
        f"early_stopping_patience={args.early_stopping_patience} | "
        f"ema_decay={args.ema_decay:.6g} | "
        f"ema_eval={int(bool(args.ema_eval))}",
        flush=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_direction_model.pt")
    best_train_path = os.path.join(args.save_dir, "best_train_direction_model.pt")
    last_path = os.path.join(args.save_dir, "last_direction_model.pt")
    best_val_loss = float("inf")
    best_checkpoint_value = None
    best_checkpoint_epoch = 0
    best_train_direction_accuracy = float("-inf")
    best_train_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    step_rows = []
    epoch_rows = []
    global_step = 0
    dashboard = {}

    for epoch in range(args.num_epochs):
        train_dataset.set_epoch_seed(epoch)

        train_loss, train_da, global_step = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            global_step_start=global_step,
            step_rows=step_rows,
            log_interval=args.log_interval,
            scheduler=scheduler,
            scheduler_step_mode=scheduler_step_mode,
            ema_helper=ema_helper,
            ema_start_step=args.ema_start_step,
        )

        ema_weights_applied = False
        if (
            ema_helper is not None
            and bool(args.ema_eval)
            and ema_helper.has_state()
            and global_step >= int(args.ema_start_step)
        ):
            ema_helper.store(model)
            ema_helper.copy_to(model)
            ema_weights_applied = True

        val_loss, da = evaluate(
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None and scheduler_step_mode == "epoch":
            plateau_metric = (
                da if str(args.plateau_metric).strip().lower() == "val_direction_accuracy" else val_loss
            )
            scheduler.step(plateau_metric)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        current_checkpoint_value = da if checkpoint_metric == "val_direction_accuracy" else val_loss
        checkpoint_improved = _metric_improved(
            current_checkpoint_value,
            best_checkpoint_value,
            higher_is_better=higher_is_better,
            min_delta=float(args.early_stopping_min_delta),
        )

        if checkpoint_improved:
            best_checkpoint_value = float(current_checkpoint_value)
            best_checkpoint_epoch = int(epoch + 1)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Direction Acc (MDA): {train_da:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Direction Acc (MDA): {da:.4f} | "
            f"LR: {current_lr:.6g} | "
            f"Eval EMA: {int(ema_weights_applied)}"
        )

        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "train_direction_accuracy": float(train_da),
                "val_loss": float(val_loss),
                "val_direction_accuracy": float(da),
                "lr": float(current_lr),
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_metric_value": float(current_checkpoint_value),
                "eval_used_ema": bool(ema_weights_applied),
            }
        )
        epoch_rows.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "train_direction_accuracy_pct": float(train_da * 100.0),
                "val_loss": float(val_loss),
                "val_direction_accuracy_pct": float(da * 100.0),
                "lr": float(current_lr),
                "checkpoint_metric_value": float(current_checkpoint_value),
                "best_checkpoint_value_so_far": float(best_checkpoint_value),
                "eval_used_ema": float(1.0 if ema_weights_applied else 0.0),
                "best_val_loss_so_far": float(best_val_loss),
            }
        )
        dashboard = save_training_dashboard(
            out_dir=args.save_dir,
            prefix="direction",
            step_rows=step_rows,
            epoch_rows=epoch_rows,
        )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "tokenizer_path": args.pretrained_tokenizer_path,
            "predictor_path": args.pretrained_predictor_path,
            "config": vars(args),
            **param_setup,
            "best_val_loss": float(best_val_loss),
            "best_checkpoint_metric": checkpoint_metric,
            "best_checkpoint_value": float(best_checkpoint_value),
            "best_epoch": int(best_checkpoint_epoch),
            "eval_used_ema": bool(ema_weights_applied),
        }

        if checkpoint_improved:
            torch.save(checkpoint_payload, best_path)

        if bool(args.save_best_train) and train_da > best_train_direction_accuracy:
            best_train_direction_accuracy = float(train_da)
            best_train_epoch = int(epoch + 1)
            train_payload = dict(checkpoint_payload)
            train_payload["best_train_direction_accuracy"] = float(best_train_direction_accuracy)
            train_payload["best_train_epoch"] = int(best_train_epoch)
            torch.save(train_payload, best_train_path)

        if ema_weights_applied:
            ema_helper.restore(model)

        if bool(args.save_last):
            last_payload = {
                "model_state_dict": model.state_dict(),
                "tokenizer_path": args.pretrained_tokenizer_path,
                "predictor_path": args.pretrained_predictor_path,
                "config": vars(args),
                **param_setup,
                "epoch": int(epoch + 1),
                "train_direction_accuracy": float(train_da),
                "val_direction_accuracy": float(da),
                "val_loss": float(val_loss),
            }
            torch.save(last_payload, last_path)

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                "Early stopping triggered | "
                f"patience={args.early_stopping_patience} | "
                f"best_epoch={best_checkpoint_epoch} | "
                f"best_{checkpoint_metric}={best_checkpoint_value:.6f}",
                flush=True,
            )
            break

    metrics_payload = {
        "best_checkpoint_path": best_path,
        "best_val_loss": float(best_val_loss),
        "best_checkpoint_metric": checkpoint_metric,
        "best_checkpoint_value": float(best_checkpoint_value) if best_checkpoint_value is not None else None,
        "best_checkpoint_epoch": int(best_checkpoint_epoch),
        "best_train_checkpoint_path": best_train_path if bool(args.save_best_train) else "",
        "best_train_direction_accuracy": (
            float(best_train_direction_accuracy)
            if best_train_direction_accuracy > float("-inf")
            else None
        ),
        "best_train_epoch": int(best_train_epoch),
        "last_checkpoint_path": last_path if bool(args.save_last) else "",
        "stopped_early": bool(stopped_early),
        "epochs_completed": int(len(history)),
        "epochs": history,
        "param_setup": param_setup,
        "config": vars(args),
        "dashboard": dashboard,
    }
    metrics_path = (
        args.save_metrics_json
        if args.save_metrics_json
        else os.path.join(args.save_dir, "direction_train_metrics.json")
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved direction metrics: {metrics_path}")


if __name__ == "__main__":
    main()
