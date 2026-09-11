#!/usr/bin/env python3
"""Build cleaner report-ready training-curve candidates from saved metric CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(".")
OUT_DIR = ROOT / "reports" / "sgp2_extension_20260405"


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, float]] = []
    for row in rows:
        casted: dict[str, float] = {}
        for key, value in row.items():
            try:
                casted[key] = float(value)
            except Exception:
                pass
        out.append(casted)
    return out


def plot_5min_epoch_curves() -> Path:
    path = ROOT / "finetune_csv" / "runs_requested_by_user" / "research_5min_direction_head_oldsplit_20260318" / "checkpoints" / "direction_epoch_metrics.csv"
    rows = read_csv(path)
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    best_val = [r["best_val_loss_so_far"] for r in rows]
    train_acc = [r["train_direction_accuracy_pct"] for r in rows]
    val_acc = [r["val_direction_accuracy_pct"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180)

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_loss, label="Validation loss", linewidth=2.0, color="#ff7f0e")
    ax.plot(epochs, best_val, label="Best val loss so far", linewidth=1.6, linestyle="--", color="#2ca02c")
    ax.set_title("5-minute winner: loss by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train accuracy", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_acc, label="Validation accuracy", linewidth=2.0, color="#ff7f0e")
    ax.set_title("5-minute winner: accuracy by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Training behavior of the selected 5-minute direction-head model", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "training_curve_5min_winner_epoch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_daily_last6_retrain_curves() -> Path:
    path = ROOT / "finetune_csv" / "runs_requested_by_user" / "retrain_cons_only_lr3e6_last6_adapter_5epochs_20260319" / "metrics" / "basemodel_epoch_metrics.csv"
    rows = read_csv(path)
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_cons_dir_loss"] for r in rows]
    val_loss = [r["val_cons_dir_loss"] for r in rows]
    train_acc = [r["train_cons_dir_acc_pct"] for r in rows]
    val_acc = [r["val_cons_dir_acc_pct"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180)

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train consecutive loss", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_loss, label="Validation consecutive loss", linewidth=2.0, color="#d62728")
    ax.scatter([epochs[val_loss.index(min(val_loss))]], [min(val_loss)], color="#2ca02c", s=28, zorder=3, label="Best validation point")
    ax.set_title("Daily last-6 retrain: consecutive loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train consecutive accuracy", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_acc, label="Validation consecutive accuracy", linewidth=2.0, color="#d62728")
    ax.set_title("Daily last-6 retrain: consecutive accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Training behavior of the daily last-6 path-only family", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "training_curve_daily_last6_retrain_epoch.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_5min_epoch_curves(),
        plot_daily_last6_retrain_curves(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
