#!/usr/bin/env python3
"""End-to-end 1m research pipeline for Kronos (2017-2025 train, 2026 test).

Pipeline:
1) Split full 1m data into year-based slices.
2) Sweep base-model lookback/predict-window/loss profiles on 2017-2024 train,
   score on 2025 validation.
3) Retrain best base setup on 2017-2025, evaluate on 2026.
4) Train direction-head variants:
   - head-only linear (frozen backbone),
   - head + last N Kronos transformer layers.
5) Evaluate direction checkpoints on full 2026 and report best.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FINETUNE_STEPS = REPO_ROOT / "finetune_csv" / "run_finetune_steps.py"
RUN_DIRECTION_TRAIN = REPO_ROOT / "finetune_csv" / "finetune_direction_model.py"
RUN_DIRECTION_EVAL = REPO_ROOT / "finetune_csv" / "eval_direction_model.py"

REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
BASE_TUNE_METRIC = "directional_accuracy_candle_close_vs_open"


@dataclass
class BaseLossProfile:
    name: str
    objective_mode: str
    ce_weight: float
    dir_weight: float
    cons_dir_weight: float
    mse_weight: float


BASE_LOSS_PROFILES: list[BaseLossProfile] = [
    BaseLossProfile(
        name="ce_only",
        objective_mode="hybrid",
        ce_weight=1.0,
        dir_weight=0.0,
        cons_dir_weight=0.0,
        mse_weight=0.0,
    ),
    BaseLossProfile(
        name="ce_cons_hybrid",
        objective_mode="hybrid",
        ce_weight=1.0,
        dir_weight=0.0,
        cons_dir_weight=0.5,
        mse_weight=0.0,
    ),
    BaseLossProfile(
        name="cons_only",
        objective_mode="consecutive_path_only",
        ce_weight=0.0,
        dir_weight=0.0,
        cons_dir_weight=1.0,
        mse_weight=0.0,
    ),
]


def parse_int_grid(raw: str, field_name: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be comma-separated integers. Bad value: {token!r}") from exc
        if v <= 0:
            raise ValueError(f"{field_name} values must be > 0. Got: {v}")
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"{field_name} resolved to empty list")
    return vals


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool):
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))


def check_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")


def write_year_slice(df: pd.DataFrame, year_start: int, year_end: int, out_path: Path):
    years = df["timestamps"].dt.year
    mask = (years >= year_start) & (years <= year_end)
    sub = df.loc[mask, REQUIRED_COLUMNS].copy()
    if sub.empty:
        raise ValueError(f"No rows for year range {year_start}-{year_end} in source data.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(sub)),
        "start": str(sub["timestamps"].iloc[0]),
        "end": str(sub["timestamps"].iloc[-1]),
        "year_start": int(year_start),
        "year_end": int(year_end),
    }


def load_eval_score(eval_json_path: Path, metric_key: str = BASE_TUNE_METRIC) -> float:
    payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
    if metric_key not in payload:
        raise KeyError(f"Metric {metric_key!r} not found in {eval_json_path}")
    return float(payload[metric_key])


def maybe_load_min_val_loss(metrics_csv_path: Path) -> float | None:
    if not metrics_csv_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_csv_path)
    except Exception:
        return None
    if "val_loss" not in df.columns or df.empty:
        return None
    return float(df["val_loss"].min())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_child_device_arg(device_arg: str) -> str:
    normalized = str(device_arg).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return "cuda:0"
    return device_arg


def main():
    parser = argparse.ArgumentParser(description="1m Kronos research pipeline.")
    parser.add_argument("--full-data-path", type=str, required=True, help="Full 1m CSV containing 2017-2026.")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="",
        help="Output root directory. Default: finetune_csv/runs_requested_by_user/research_1m_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device hint (e.g., cuda:0 or cpu).")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--train-start-year", type=int, default=2017)
    parser.add_argument("--tune-train-end-year", type=int, default=2024)
    parser.add_argument("--tune-val-year", type=int, default=2025)
    parser.add_argument("--final-train-end-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)

    parser.add_argument("--lookbacks", type=str, default="256,512", help="Base-model lookback candidates.")
    parser.add_argument("--predict-windows", type=str, default="16", help="Base-model predict-window candidates.")
    parser.add_argument("--max-base-trials", type=int, default=6, help="Cap number of base-model trials (<=0 = no cap).")

    parser.add_argument("--tokenizer-epochs", type=int, default=4)
    parser.add_argument("--basemodel-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument(
        "--tokenizer-pretrained",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
    )
    parser.add_argument(
        "--predictor-pretrained",
        type=str,
        default="NeoQuasar/Kronos-base",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tokenizer training in base-model trials.",
    )
    parser.add_argument(
        "--auto-tune-max-trials",
        type=int,
        default=6,
        help="Max eval lookback/block_len combos per base trial (<=0 = no cap).",
    )

    parser.add_argument("--direction-lookbacks", type=str, default="256,512")
    parser.add_argument("--direction-epochs", type=int, default=8)
    parser.add_argument("--direction-batch-size", type=int, default=64)
    parser.add_argument("--direction-learning-rate", type=float, default=1e-4)
    parser.add_argument("--direction-vol-window", type=int, default=1440)
    parser.add_argument("--direction-vol-k", type=float, default=0.25)
    parser.add_argument("--direction-min-eps", type=float, default=0.0)
    parser.add_argument("--direction-unfreeze-last-n", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_work_dir = REPO_ROOT / "finetune_csv" / "runs_requested_by_user" / f"research_1m_{ts}"
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work_dir
    data_dir = work_dir / "data"
    configs_dir = work_dir / "configs"
    logs_dir = work_dir / "logs"
    out_finetuned = work_dir / "finetuned"
    summary_path = work_dir / "research_summary.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_finetuned.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.device.startswith("cuda:"):
        gpu_idx = args.device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    elif args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    child_device = normalize_child_device_arg(args.device)

    full_path = Path(args.full_data_path).expanduser().resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Full data CSV not found: {full_path}")

    print("Loading and splitting yearly datasets...", flush=True)
    df_full = pd.read_csv(full_path)
    check_required_columns(df_full)
    df_full["timestamps"] = pd.to_datetime(df_full["timestamps"])
    df_full = df_full.sort_values("timestamps").drop_duplicates("timestamps", keep="last").reset_index(drop=True)

    split_tune_train = write_year_slice(
        df=df_full,
        year_start=args.train_start_year,
        year_end=args.tune_train_end_year,
        out_path=data_dir / f"BTCUSDT_kline_1m_{args.train_start_year}_{args.tune_train_end_year}_tune_train.csv",
    )
    split_tune_val = write_year_slice(
        df=df_full,
        year_start=args.tune_val_year,
        year_end=args.tune_val_year,
        out_path=data_dir / f"BTCUSDT_kline_1m_{args.tune_val_year}_tune_val.csv",
    )
    split_final_train = write_year_slice(
        df=df_full,
        year_start=args.train_start_year,
        year_end=args.final_train_end_year,
        out_path=data_dir / f"BTCUSDT_kline_1m_{args.train_start_year}_{args.final_train_end_year}_final_train.csv",
    )
    split_test = write_year_slice(
        df=df_full,
        year_start=args.test_year,
        year_end=args.test_year,
        out_path=data_dir / f"BTCUSDT_kline_1m_{args.test_year}_test.csv",
    )

    lookbacks = parse_int_grid(args.lookbacks, "--lookbacks")
    predict_windows = parse_int_grid(args.predict_windows, "--predict-windows")
    direction_lookbacks = parse_int_grid(args.direction_lookbacks, "--direction-lookbacks")

    base_trials: list[dict[str, Any]] = []
    trial_count = 0
    for lb in lookbacks:
        for pw in predict_windows:
            for profile in BASE_LOSS_PROFILES:
                if args.max_base_trials > 0 and trial_count >= args.max_base_trials:
                    break
                trial_count += 1

                exp_name = f"BTCUSDT_kline_1m_tune_lb{lb}_pw{pw}_{profile.name}"
                config_path = configs_dir / f"{exp_name}.yaml"
                metrics_dir = work_dir / "metrics" / exp_name
                eval_prefix = "val2025_actual"

                cmd = [
                    sys.executable,
                    str(RUN_FINETUNE_STEPS),
                    "--data-path",
                    split_tune_train["path"],
                    "--config-path",
                    str(config_path),
                    "--exp-name",
                    exp_name,
                    "--base-path",
                    str(out_finetuned),
                    "--tokenizer-pretrained",
                    args.tokenizer_pretrained,
                    "--predictor-pretrained",
                    args.predictor_pretrained,
                    "--lookback-window",
                    str(lb),
                    "--predict-window",
                    str(pw),
                    "--max-context",
                    str(lb),
                    "--clip",
                    str(args.clip),
                    "--train-ratio",
                    str(args.train_ratio),
                    "--val-ratio",
                    str(args.val_ratio),
                    "--test-ratio",
                    "0",
                    "--tokenizer-epochs",
                    str(args.tokenizer_epochs),
                    "--basemodel-epochs",
                    str(args.basemodel_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.seed),
                    "--predictor-ce-weight",
                    str(profile.ce_weight),
                    "--predictor-directional-weight",
                    str(profile.dir_weight),
                    "--predictor-consecutive-directional-weight",
                    str(profile.cons_dir_weight),
                    "--predictor-mse-weight",
                    str(profile.mse_weight),
                    "--predictor-objective-mode",
                    profile.objective_mode,
                    "--train-sampling-strategy",
                    "independent",
                    "--val-sampling-strategy",
                    "consecutive",
                    "--metrics-dir",
                    str(metrics_dir),
                    "--run-consecutive-eval",
                    "--eval-path",
                    split_tune_val["path"],
                    "--eval-feedback-source",
                    "actual",
                    "--eval-lookback",
                    str(lb),
                    "--eval-block-len",
                    str(pw),
                    "--eval-output-prefix",
                    eval_prefix,
                    "--auto-tune-eval-hparams",
                    "--auto-tune-metric",
                    BASE_TUNE_METRIC,
                    "--auto-tune-max-trials",
                    str(args.auto_tune_max_trials),
                ]
                if args.skip_tokenizer:
                    cmd.append("--skip-tokenizer")
                if args.device == "cpu":
                    cmd.append("--force-cpu")

                run_cmd(cmd, env=env, dry_run=args.dry_run)

                eval_json = out_finetuned / exp_name / "eval" / f"{eval_prefix}.json"
                epoch_metrics_csv = metrics_dir / "basemodel_epoch_metrics.csv"
                if args.dry_run:
                    score = float("nan")
                    min_val_loss = None
                else:
                    score = load_eval_score(eval_json, metric_key=BASE_TUNE_METRIC)
                    min_val_loss = maybe_load_min_val_loss(epoch_metrics_csv)

                base_trials.append(
                    {
                        "exp_name": exp_name,
                        "lookback": lb,
                        "predict_window": pw,
                        "loss_profile": profile.name,
                        "objective_mode": profile.objective_mode,
                        "ce_weight": profile.ce_weight,
                        "dir_weight": profile.dir_weight,
                        "cons_dir_weight": profile.cons_dir_weight,
                        "mse_weight": profile.mse_weight,
                        "eval_json": str(eval_json),
                        "eval_metric_key": BASE_TUNE_METRIC,
                        "eval_metric_value": score,
                        "min_train_val_loss": min_val_loss,
                    }
                )
            if args.max_base_trials > 0 and trial_count >= args.max_base_trials:
                break
        if args.max_base_trials > 0 and trial_count >= args.max_base_trials:
            break

    if not base_trials:
        raise RuntimeError("No base trials executed.")

    if args.dry_run:
        best_base = base_trials[0]
    else:
        best_base = max(
            base_trials,
            key=lambda x: (
                float(x["eval_metric_value"]),
                -float(x["min_train_val_loss"]) if x["min_train_val_loss"] is not None else -1e9,
            ),
        )

    best_lb = int(best_base["lookback"])
    best_pw = int(best_base["predict_window"])
    best_profile = next(p for p in BASE_LOSS_PROFILES if p.name == best_base["loss_profile"])

    final_exp_name = f"BTCUSDT_kline_1m_final_lb{best_lb}_pw{best_pw}_{best_profile.name}"
    final_config_path = configs_dir / f"{final_exp_name}.yaml"
    final_metrics_dir = work_dir / "metrics" / final_exp_name
    final_eval_prefix = "test2026_actual"

    final_cmd = [
        sys.executable,
        str(RUN_FINETUNE_STEPS),
        "--data-path",
        split_final_train["path"],
        "--config-path",
        str(final_config_path),
        "--exp-name",
        final_exp_name,
        "--base-path",
        str(out_finetuned),
        "--tokenizer-pretrained",
        args.tokenizer_pretrained,
        "--predictor-pretrained",
        args.predictor_pretrained,
        "--lookback-window",
        str(best_lb),
        "--predict-window",
        str(best_pw),
        "--max-context",
        str(best_lb),
        "--clip",
        str(args.clip),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        "0",
        "--tokenizer-epochs",
        str(args.tokenizer_epochs),
        "--basemodel-epochs",
        str(args.basemodel_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--predictor-ce-weight",
        str(best_profile.ce_weight),
        "--predictor-directional-weight",
        str(best_profile.dir_weight),
        "--predictor-consecutive-directional-weight",
        str(best_profile.cons_dir_weight),
        "--predictor-mse-weight",
        str(best_profile.mse_weight),
        "--predictor-objective-mode",
        best_profile.objective_mode,
        "--train-sampling-strategy",
        "independent",
        "--val-sampling-strategy",
        "consecutive",
        "--metrics-dir",
        str(final_metrics_dir),
        "--run-consecutive-eval",
        "--eval-path",
        split_test["path"],
        "--eval-feedback-source",
        "actual",
        "--eval-lookback",
        str(best_lb),
        "--eval-block-len",
        str(best_pw),
        "--eval-output-prefix",
        final_eval_prefix,
    ]
    if args.skip_tokenizer:
        final_cmd.append("--skip-tokenizer")
    if args.device == "cpu":
        final_cmd.append("--force-cpu")

    run_cmd(final_cmd, env=env, dry_run=args.dry_run)

    final_eval_json = out_finetuned / final_exp_name / "eval" / f"{final_eval_prefix}.json"
    final_predictor_path = out_finetuned / final_exp_name / "basemodel" / "best_model"
    final_tokenizer_path = out_finetuned / final_exp_name / "tokenizer" / "best_model"
    if args.skip_tokenizer and not final_tokenizer_path.exists():
        final_tokenizer_path = Path(args.tokenizer_pretrained)

    direction_trials: list[dict[str, Any]] = []
    direction_variants = [
        {
            "name": "head_only_linear",
            "freeze_backbone": True,
            "unfreeze_last_n": 0,
            "head_hidden_dim": 0,
        },
        {
            "name": "head_plus_lastn",
            "freeze_backbone": True,
            "unfreeze_last_n": int(args.direction_unfreeze_last_n),
            "head_hidden_dim": 512,
        },
    ]

    for lb in direction_lookbacks:
        for variant in direction_variants:
            run_name = f"{variant['name']}_lb{lb}"
            run_dir = work_dir / "direction_runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            train_metrics_json = run_dir / "train_metrics.json"
            train_cmd = [
                sys.executable,
                str(RUN_DIRECTION_TRAIN),
                "--data-path",
                split_final_train["path"],
                "--pretrained-tokenizer-path",
                str(final_tokenizer_path),
                "--pretrained-predictor-path",
                str(final_predictor_path),
                "--lookback-window",
                str(lb),
                "--horizon-steps",
                "1",
                "--vol-window",
                str(args.direction_vol_window),
                "--vol-k",
                str(args.direction_vol_k),
                "--min-eps",
                str(args.direction_min_eps),
                "--batch-size",
                str(args.direction_batch_size),
                "--num-epochs",
                str(args.direction_epochs),
                "--learning-rate",
                str(args.direction_learning_rate),
                "--clip",
                str(args.clip),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--device",
                child_device,
                "--num-workers",
                str(args.num_workers),
                "--save-dir",
                str(run_dir),
                "--freeze-backbone" if variant["freeze_backbone"] else "--no-freeze-backbone",
                "--unfreeze-last-n-transformer-layers",
                str(variant["unfreeze_last_n"]),
                "--head-hidden-dim",
                str(variant["head_hidden_dim"]),
                "--head-dropout",
                "0.1",
                "--head-use-layernorm",
                "--head-pooling",
                "last",
                "--loss-type",
                "bce",
                "--save-metrics-json",
                str(train_metrics_json),
            ]
            run_cmd(train_cmd, env=env, dry_run=args.dry_run)

            ckpt_path = run_dir / "best_direction_model.pt"
            eval_json = run_dir / "eval_2026.json"
            eval_csv = run_dir / "pred_2026.csv"
            eval_cmd = [
                sys.executable,
                str(RUN_DIRECTION_EVAL),
                "--data-path",
                split_test["path"],
                "--checkpoint-path",
                str(ckpt_path),
                "--tokenizer-path",
                str(final_tokenizer_path),
                "--predictor-path",
                str(final_predictor_path),
                "--data-type",
                "test",
                "--lookback-window",
                str(lb),
                "--horizon-steps",
                "1",
                "--vol-window",
                str(args.direction_vol_window),
                "--vol-k",
                str(args.direction_vol_k),
                "--min-eps",
                str(args.direction_min_eps),
                "--clip",
                str(args.clip),
                "--train-ratio",
                "0",
                "--val-ratio",
                "0",
                "--test-ratio",
                "1",
                "--batch-size",
                str(max(128, args.direction_batch_size)),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--device",
                child_device,
                "--output-json",
                str(eval_json),
                "--output-csv",
                str(eval_csv),
            ]
            run_cmd(eval_cmd, env=env, dry_run=args.dry_run)

            if args.dry_run:
                direction_acc = float("nan")
                direction_loss = float("nan")
                best_val_loss = float("nan")
            else:
                eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
                train_payload = json.loads(train_metrics_json.read_text(encoding="utf-8"))
                direction_acc = float(eval_payload.get("directional_accuracy_masked", float("nan")))
                direction_loss = float(eval_payload.get("avg_bce_loss_masked", float("nan")))
                best_val_loss = float(train_payload.get("best_val_loss", float("nan")))

            direction_trials.append(
                {
                    "run_name": run_name,
                    "variant": variant["name"],
                    "lookback": lb,
                    "freeze_backbone": variant["freeze_backbone"],
                    "unfreeze_last_n": variant["unfreeze_last_n"],
                    "head_hidden_dim": variant["head_hidden_dim"],
                    "train_metrics_json": str(train_metrics_json),
                    "eval_json": str(eval_json),
                    "eval_csv": str(eval_csv),
                    "eval_directional_accuracy_masked": direction_acc,
                    "eval_avg_bce_loss_masked": direction_loss,
                    "train_best_val_loss": best_val_loss,
                }
            )

    if args.dry_run:
        best_direction = direction_trials[0]
        final_base_eval_metric = float("nan")
        final_base_eval_payload = {}
    else:
        best_direction = max(
            direction_trials,
            key=lambda x: (
                float(x["eval_directional_accuracy_masked"]),
                -float(x["eval_avg_bce_loss_masked"]),
            ),
        )
        final_base_eval_payload = load_json(final_eval_json)
        final_base_eval_metric = float(final_base_eval_payload[BASE_TUNE_METRIC])

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "inputs": {
            "full_data_path": str(full_path),
            "device": args.device,
            "seed": args.seed,
        },
        "year_splits": {
            "tune_train": split_tune_train,
            "tune_val": split_tune_val,
            "final_train": split_final_train,
            "test": split_test,
        },
        "base_tuning_metric": BASE_TUNE_METRIC,
        "base_trials": base_trials,
        "best_base_trial": best_base,
        "final_base_run": {
            "exp_name": final_exp_name,
            "eval_json": str(final_eval_json),
            "eval_metric_value": final_base_eval_metric,
            "eval_metrics": final_base_eval_payload,
            "predictor_path": str(final_predictor_path),
            "tokenizer_path": str(final_tokenizer_path),
        },
        "direction_trials": direction_trials,
        "best_direction_trial": best_direction,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
