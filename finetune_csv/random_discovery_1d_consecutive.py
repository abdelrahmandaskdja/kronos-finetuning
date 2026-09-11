#!/usr/bin/env python3
"""Random-search discovery for 1D BTC consecutive directional accuracy on 2026."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = REPO_ROOT / "finetune_csv"

TRAIN_DATA = FINETUNE_DIR / "data" / "BTCUSDT_kline_1d_2018_2024.csv"
EVAL_DATA = FINETUNE_DIR / "data" / "BTCUSDT_kline_1d_2026_to_now.csv"

LOCAL_1D_TOKENIZER = FINETUNE_DIR / "finetuned" / "BTCUSDT_kline_1d_2018_2025" / "tokenizer" / "best_model"
LOCAL_1D_PREDICTOR = FINETUNE_DIR / "finetuned" / "BTCUSDT_kline_1d_2018_2025" / "basemodel" / "best_model"
LOCAL_5M_PREDICTOR = FINETUNE_DIR / "finetuned" / "BTCUSDT_kline_5min_2018_2025" / "basemodel" / "best_model"
LOCAL_5M_TOKENIZER = FINETUNE_DIR / "finetuned" / "BTCUSDT_kline_5min_2018_2025" / "tokenizer" / "best_model"

METRIC_KEY = "directional_accuracy_close_vs_prev_close_consecutive_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random discovery for 1D consecutive metric")
    parser.add_argument("--trials", type=int, default=12, help="Number of random trials")
    parser.add_argument("--device", type=str, default="cuda:1", help='Training/eval device, e.g. "cuda:1" or "cpu"')
    parser.add_argument("--gpu-index", type=str, default="1", help='CUDA_VISIBLE_DEVICES index, e.g. "1"')
    parser.add_argument("--seed", type=int, default=20260225, help="Random seed for search reproducibility")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FINETUNE_DIR / "discovery_1d_consecutive",
        help="Output directory for configs/logs/results",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True, stdout=f, stderr=subprocess.STDOUT)


def sample_trial(idx: int, rng: random.Random) -> dict[str, Any]:
    lookback = rng.choice([32, 64, 96, 128, 160, 192])
    predict_window = rng.choice([2, 4, 8, 16, 24])
    if lookback < predict_window + 16:
        lookback = predict_window + 16
    # Validation split is 10% of 2557 rows (~256). Keep window comfortably below that.
    max_window = 240
    if lookback + predict_window + 1 > max_window:
        lookback = max(32, max_window - predict_window - 1)

    return {
        "trial_id": idx,
        "lookback_window": lookback,
        "predict_window": predict_window,
        "batch_size": rng.choice([1, 2, 4]),
        "num_workers": rng.choice([0, 1, 2]),
        "epochs": rng.choice([3, 4, 5, 6]),
        "predictor_lr": rng.choice([3e-7, 1e-6, 3e-6, 1e-5]),
        "ce_weight": rng.choice([0.25, 0.5, 1.0]),
        "dir_weight": rng.choice([0.0, 0.25, 0.5]),
        "cons_dir_weight": rng.choice([0.5, 1.0, 2.0, 4.0, 8.0]),
        "mse_weight": rng.choice([0.0, 0.001, 0.005, 0.01]),
        "cons_pos_weight": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
        "cons_focal_gamma": rng.choice([0.0, 0.5, 1.0, 1.5, 2.0]),
        "cons_label_smoothing": rng.choice([0.0, 0.02, 0.05]),
        "pretrained_predictor": rng.choice([str(LOCAL_1D_PREDICTOR), str(LOCAL_5M_PREDICTOR)]),
        "pretrained_tokenizer": rng.choice([str(LOCAL_1D_TOKENIZER), str(LOCAL_5M_TOKENIZER)]),
        "seed": rng.randint(1, 10_000_000),
    }


def make_config(trial: dict[str, Any], exp_name: str, use_cuda: bool, device_id: int) -> dict[str, Any]:
    return {
        "data": {
            "data_path": str(TRAIN_DATA),
            "lookback_window": int(trial["lookback_window"]),
            "predict_window": int(trial["predict_window"]),
            "max_context": 512,
            "clip": 5.0,
            "train_ratio": 0.9,
            "val_ratio": 0.1,
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": 1,
            "basemodel_epochs": int(trial["epochs"]),
            "batch_size": int(trial["batch_size"]),
            "log_interval": 20,
            "num_workers": int(trial["num_workers"]),
            "seed": int(trial["seed"]),
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": float(trial["predictor_lr"]),
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
            "predictor_ce_loss_weight": float(trial["ce_weight"]),
            "predictor_directional_loss_weight": float(trial["dir_weight"]),
            "predictor_consecutive_directional_loss_weight": float(trial["cons_dir_weight"]),
            "predictor_consecutive_pos_weight": float(trial["cons_pos_weight"]),
            "predictor_consecutive_focal_gamma": float(trial["cons_focal_gamma"]),
            "predictor_consecutive_label_smoothing": float(trial["cons_label_smoothing"]),
            "predictor_mse_loss_weight": float(trial["mse_weight"]),
        },
        "model_paths": {
            "pretrained_tokenizer": str(trial["pretrained_tokenizer"]),
            "pretrained_predictor": str(trial["pretrained_predictor"]),
            "exp_name": exp_name,
            "base_path": str(FINETUNE_DIR / "finetuned"),
            "base_save_path": "",
            "finetuned_tokenizer": str(trial["pretrained_tokenizer"]),
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": "kronos_1d_consecutive_discovery",
            "description": "Random search trial for 1D consecutive directional metric",
            "use_comet": False,
            "train_tokenizer": False,
            "train_basemodel": True,
            "skip_existing": False,
            "pre_trained_tokenizer": True,
            "pre_trained_predictor": True,
        },
        "device": {
            "use_cuda": bool(use_cuda),
            "device_id": int(device_id),
        },
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    res_path = out_dir / "results.jsonl"
    best_path = out_dir / "best_result.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    use_cuda = device.startswith("cuda")
    device_id = 0
    if ":" in device:
        try:
            device_id = int(device.split(":", 1)[1])
        except ValueError:
            device_id = 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if use_cuda:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    eval_device = "cuda:0" if use_cuda else device

    best_metric = float("-inf")
    best_result: dict[str, Any] | None = None

    for i in range(1, args.trials + 1):
        trial = sample_trial(i, rng)
        exp_name = f"BTCUSDT_kline_1d_2018_2024_discovery_t{i:03d}"
        config = make_config(trial, exp_name=exp_name, use_cuda=use_cuda, device_id=device_id)

        cfg_path = cfg_dir / f"{exp_name}.yaml"
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        train_log = log_dir / f"{exp_name}_train.log"
        eval_json = out_dir / f"{exp_name}_eval_2026.json"
        eval_csv = out_dir / f"{exp_name}_pred_2026.csv"
        eval_log = log_dir / f"{exp_name}_eval.log"

        start_ts = time.time()
        status = "ok"
        err = ""

        model_id = FINETUNE_DIR / "finetuned" / exp_name / "basemodel" / "best_model"
        tokenizer_id = Path(config["model_paths"]["finetuned_tokenizer"])

        try:
            run_cmd(
                [os.sys.executable, "finetune_base_model.py", "--config", str(cfg_path)],
                cwd=FINETUNE_DIR,
                env=env,
                log_path=train_log,
            )
            if not (model_id / "model.safetensors").exists():
                raise RuntimeError(f"Missing trained model checkpoint: {model_id}")

            run_cmd(
                [
                    os.sys.executable,
                    "eval_consecutive_rollout.py",
                    "--train-context-path",
                    str(TRAIN_DATA),
                    "--eval-path",
                    str(EVAL_DATA),
                    "--model-id",
                    str(model_id),
                    "--tokenizer-id",
                    str(tokenizer_id),
                    "--lookback",
                    str(int(trial["lookback_window"])),
                    "--block-len",
                    str(max(2, min(16, int(trial["predict_window"])))),
                    "--device",
                    eval_device,
                    "--output-csv",
                    str(eval_csv),
                    "--output-json",
                    str(eval_json),
                ],
                cwd=FINETUNE_DIR,
                env=env,
                log_path=eval_log,
            )
        except Exception as exc:
            status = "failed"
            err = str(exc)

        elapsed = time.time() - start_ts
        metrics: dict[str, Any] = {}
        metric_val = None
        if status == "ok" and eval_json.exists():
            metrics = json.loads(eval_json.read_text(encoding="utf-8"))
            metric_val = float(metrics.get(METRIC_KEY, float("nan")))
            if metric_val > best_metric:
                best_metric = metric_val
                best_result = {
                    "trial": trial,
                    "exp_name": exp_name,
                    "metric_key": METRIC_KEY,
                    "metric_value": metric_val,
                    "model_id": str(model_id),
                    "tokenizer_id": str(tokenizer_id),
                    "eval_json": str(eval_json),
                    "eval_csv": str(eval_csv),
                    "train_log": str(train_log),
                    "eval_log": str(eval_log),
                }
                best_path.write_text(json.dumps(best_result, indent=2), encoding="utf-8")

        row = {
            "trial_index": i,
            "exp_name": exp_name,
            "status": status,
            "error": err,
            "elapsed_seconds": elapsed,
            "metric_key": METRIC_KEY,
            "metric_value": metric_val,
            "trial": trial,
            "eval_json": str(eval_json),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
            "metrics": metrics if metrics else None,
        }
        with res_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[trial {i}/{args.trials}] status={status} metric={metric_val} "
            f"elapsed={elapsed:.1f}s exp={exp_name}",
            flush=True,
        )
        if best_result is not None:
            print(
                f"  best_so_far={best_result['metric_value']:.6f} "
                f"exp={best_result['exp_name']}",
                flush=True,
            )

    if best_result is None:
        raise SystemExit("No successful trial completed.")
    print("\nBest trial:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()
