import csv
import json
import os
from pathlib import Path
from typing import Any


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = row.get(key, None)
        if val is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _plot_lines(
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    title: str,
    y_label: str,
    out_path: Path,
    logger=None,
) -> bool:
    if not rows or not y_keys:
        return False
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning(f"Matplotlib unavailable, skipping chart {out_path.name}: {exc}")
        return False

    x = _to_float_list(rows, x_key)
    plt.figure(figsize=(12, 6))
    plotted = 0
    for key in y_keys:
        y = _to_float_list(rows, key)
        if len(y) != len(x):
            continue
        plt.plot(x, y, label=key)
        plotted += 1
    if plotted == 0:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def save_training_dashboard(
    out_dir: str | Path,
    prefix: str,
    step_rows: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    logger=None,
) -> dict[str, str]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    step_csv = out_root / f"{prefix}_step_metrics.csv"
    epoch_csv = out_root / f"{prefix}_epoch_metrics.csv"
    _write_csv(step_rows, step_csv)
    _write_csv(epoch_rows, epoch_csv)

    step_loss_png = out_root / f"{prefix}_step_loss.png"
    step_acc_png = out_root / f"{prefix}_step_accuracy_pct.png"
    epoch_loss_png = out_root / f"{prefix}_epoch_loss.png"
    epoch_acc_png = out_root / f"{prefix}_epoch_accuracy_pct.png"

    step_loss_keys: list[str] = []
    step_acc_keys: list[str] = []
    if step_rows:
        for key in step_rows[0].keys():
            key_l = key.lower()
            if "loss" in key_l:
                step_loss_keys.append(key)
            if (("acc" in key_l) or ("accuracy" in key_l)) and key_l.endswith("_pct"):
                step_acc_keys.append(key)

    epoch_loss_keys: list[str] = []
    epoch_acc_keys: list[str] = []
    if epoch_rows:
        for key in epoch_rows[0].keys():
            key_l = key.lower()
            if "loss" in key_l:
                epoch_loss_keys.append(key)
            if (("acc" in key_l) or ("accuracy" in key_l)) and key_l.endswith("_pct"):
                epoch_acc_keys.append(key)

    _plot_lines(
        step_rows,
        x_key="global_step",
        y_keys=step_loss_keys,
        title=f"{prefix} Step Loss",
        y_label="Loss",
        out_path=step_loss_png,
        logger=logger,
    )
    _plot_lines(
        step_rows,
        x_key="global_step",
        y_keys=step_acc_keys,
        title=f"{prefix} Step Accuracy (%)",
        y_label="Accuracy (%)",
        out_path=step_acc_png,
        logger=logger,
    )
    _plot_lines(
        epoch_rows,
        x_key="epoch",
        y_keys=epoch_loss_keys,
        title=f"{prefix} Epoch Loss",
        y_label="Loss",
        out_path=epoch_loss_png,
        logger=logger,
    )
    _plot_lines(
        epoch_rows,
        x_key="epoch",
        y_keys=epoch_acc_keys,
        title=f"{prefix} Epoch Accuracy (%)",
        y_label="Accuracy (%)",
        out_path=epoch_acc_png,
        logger=logger,
    )

    summary = {
        "step_csv": str(step_csv),
        "epoch_csv": str(epoch_csv),
        "step_loss_chart": str(step_loss_png),
        "step_accuracy_chart": str(step_acc_png),
        "epoch_loss_chart": str(epoch_loss_png),
        "epoch_accuracy_chart": str(epoch_acc_png),
        "step_rows": int(len(step_rows)),
        "epoch_rows": int(len(epoch_rows)),
    }
    summary_path = out_root / f"{prefix}_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    if logger is not None:
        logger.info(f"Saved {prefix} metrics dashboard: {summary_path}")
    return summary
