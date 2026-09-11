#!/usr/bin/env python3
"""
Step-by-step Kronos fine-tuning runner for custom CSV data.

This script follows the workflow documented in finetune_csv/README.md:
1) Prepare config
2) Fine-tune tokenizer
3) Fine-tune predictor (basemodel)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_COLUMNS = {"timestamps", "open", "high", "low", "close", "volume", "amount"}
EVAL_ACCURACY_KEYS = (
    "directional_accuracy_close_vs_prev_actual_close",
    "directional_accuracy_close_vs_prev_close_consecutive_path",
    "directional_accuracy_candle_close_vs_open",
    "return_weighted_directional_accuracy_close_vs_prev_actual_close",
    "return_weighted_directional_accuracy_close_vs_prev_close_consecutive_path",
)
DEFAULT_AUTO_TUNE_METRIC = "directional_accuracy_close_vs_prev_close_consecutive_path"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "finetune_csv" / "data" / "BTCUSDT_kline_5min_last6m.csv"
    default_config = repo_root / "finetune_csv" / "configs" / "config_btcusdt_5min_last6m.yaml"
    default_base_path = repo_root / "finetune_csv" / "finetuned"

    parser = argparse.ArgumentParser(description="Run Kronos fine-tuning in explicit steps")
    parser.add_argument("--data-path", type=str, default=str(default_data), help="Path to BTC kline CSV")
    parser.add_argument("--config-path", type=str, default=str(default_config), help="Generated config output path")
    parser.add_argument("--exp-name", type=str, default="BTCUSDT_kline_5min_last6m", help="Experiment name")
    parser.add_argument("--base-path", type=str, default=str(default_base_path), help="Base output dir")

    parser.add_argument(
        "--tokenizer-pretrained",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer model path or HF id",
    )
    parser.add_argument(
        "--predictor-pretrained",
        type=str,
        default="NeoQuasar/Kronos-base",
        help="Predictor model path or HF id",
    )

    parser.add_argument("--lookback-window", type=int, default=512)
    parser.add_argument("--predict-window", type=int, default=48)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)

    parser.add_argument("--tokenizer-epochs", type=int, default=30)
    parser.add_argument("--basemodel-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-lr", type=float, default=2e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-6)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--predictor-ce-weight", type=float, default=1.0)
    parser.add_argument("--predictor-directional-weight", type=float, default=0.0)
    parser.add_argument("--predictor-consecutive-directional-weight", type=float, default=0.0)
    parser.add_argument(
        "--predictor-consecutive-magnitude-mode",
        type=str,
        choices=["relative_return", "context_window"],
        default="relative_return",
        help=(
            "How to decide whether a target move is big: "
            "'relative_return' uses a fixed return scale, "
            "'context_window' compares each move against the mean absolute close move inside the sample context window."
        ),
    )
    parser.add_argument("--predictor-consecutive-magnitude-weight", type=float, default=0.0)
    parser.add_argument("--predictor-consecutive-magnitude-scale", type=float, default=0.01)
    parser.add_argument("--predictor-consecutive-magnitude-power", type=float, default=1.0)
    parser.add_argument("--predictor-consecutive-magnitude-cap", type=float, default=5.0)
    parser.add_argument("--predictor-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--predictor-objective-mode",
        type=str,
        choices=["hybrid", "consecutive_path", "consecutive_path_only"],
        default="hybrid",
        help="Loss preset for predictor objective alignment.",
    )
    parser.add_argument(
        "--train-sampling-strategy",
        type=str,
        choices=["independent", "consecutive"],
        default="independent",
    )
    parser.add_argument(
        "--val-sampling-strategy",
        type=str,
        choices=["independent", "consecutive"],
        default="consecutive",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="",
        help="Optional explicit output directory for training CSV/PNG dashboards.",
    )

    parser.add_argument(
        "--mode",
        choices=["steps", "sequential"],
        default="steps",
        help="steps: tokenizer then predictor, sequential: use train_sequential.py",
    )
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer step")
    parser.add_argument("--skip-basemodel", action="store_true", help="Skip basemodel step")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output model exists")
    parser.add_argument("--prepare-only", action="store_true", help="Only validate and generate config")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    parser.add_argument(
        "--run-consecutive-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run eval_consecutive_rollout.py after training.",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default="",
        help="Evaluation CSV path for consecutive rollout. If omitted, eval step is skipped.",
    )
    parser.add_argument(
        "--eval-feedback-source",
        type=str,
        choices=["actual", "predicted"],
        default="actual",
        help='Rollout context update source: "actual" enforces actual-data feedback.',
    )
    parser.add_argument("--eval-lookback", type=int, default=0, help="Eval lookback (<=0 uses training lookback)")
    parser.add_argument("--eval-block-len", type=int, default=0, help="Eval block length (<=0 uses training predict window)")
    parser.add_argument("--eval-output-prefix", type=str, default="", help="Optional prefix for eval output files")
    parser.add_argument(
        "--auto-tune-eval-hparams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Grid-search eval lookback/block_len on current data and select best by metric.",
    )
    parser.add_argument(
        "--auto-tune-lookbacks",
        type=str,
        default="",
        help="Comma-separated eval lookback candidates (default derives from base eval lookback).",
    )
    parser.add_argument(
        "--auto-tune-block-lens",
        type=str,
        default="",
        help="Comma-separated eval block_len candidates (default derives from base eval block_len).",
    )
    parser.add_argument(
        "--auto-tune-metric",
        type=str,
        choices=EVAL_ACCURACY_KEYS,
        default=DEFAULT_AUTO_TUNE_METRIC,
        help="Directional metric optimized by auto-tuning.",
    )
    parser.add_argument(
        "--auto-tune-max-trials",
        type=int,
        default=0,
        help="Optional hard cap on evaluated lookback/block_len combinations (<=0 means no cap).",
    )
    return parser.parse_args()


def inspect_csv(path: Path) -> Tuple[int, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        count = 0
        first_ts = ""
        last_ts = ""
        for row in reader:
            ts = row["timestamps"]
            if count == 0:
                first_ts = ts
            last_ts = ts
            count += 1

    if count == 0:
        raise ValueError(f"CSV has no rows: {path}")
    return count, first_ts, last_ts


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def float_str(value: float) -> str:
    # Emit YAML-safe decimal float text (avoid scientific notation becoming a string).
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_config_text(args: argparse.Namespace, data_path: Path, base_path: Path) -> str:
    use_cuda = not args.force_cpu
    return f"""# Auto-generated by finetune_csv/run_finetune_steps.py
data:
  data_path: "{data_path}"
  lookback_window: {args.lookback_window}
  predict_window: {args.predict_window}
  max_context: {args.max_context}
  clip: {float_str(args.clip)}
  train_ratio: {float_str(args.train_ratio)}
  val_ratio: {float_str(args.val_ratio)}
  test_ratio: {float_str(args.test_ratio)}

training:
  tokenizer_epochs: {args.tokenizer_epochs}
  basemodel_epochs: {args.basemodel_epochs}
  batch_size: {args.batch_size}
  log_interval: 50
  num_workers: {args.num_workers}
  seed: {args.seed}
  tokenizer_learning_rate: {float_str(args.tokenizer_lr)}
  predictor_learning_rate: {float_str(args.predictor_lr)}
  adam_beta1: {float_str(0.9)}
  adam_beta2: {float_str(0.95)}
  adam_weight_decay: {float_str(0.1)}
  accumulation_steps: {args.accumulation_steps}
  predictor_ce_loss_weight: {float_str(args.predictor_ce_weight)}
  predictor_directional_loss_weight: {float_str(args.predictor_directional_weight)}
  predictor_consecutive_directional_loss_weight: {float_str(args.predictor_consecutive_directional_weight)}
  predictor_consecutive_magnitude_mode: "{args.predictor_consecutive_magnitude_mode}"
  predictor_consecutive_magnitude_weight: {float_str(args.predictor_consecutive_magnitude_weight)}
  predictor_consecutive_magnitude_scale: {float_str(args.predictor_consecutive_magnitude_scale)}
  predictor_consecutive_magnitude_power: {float_str(args.predictor_consecutive_magnitude_power)}
  predictor_consecutive_magnitude_cap: {float_str(args.predictor_consecutive_magnitude_cap)}
  predictor_mse_loss_weight: {float_str(args.predictor_mse_weight)}
  predictor_objective_mode: "{args.predictor_objective_mode}"
  train_sampling_strategy: "{args.train_sampling_strategy}"
  val_sampling_strategy: "{args.val_sampling_strategy}"
  metrics_dir: "{args.metrics_dir}"

model_paths:
  pretrained_tokenizer: "{args.tokenizer_pretrained}"
  pretrained_predictor: "{args.predictor_pretrained}"
  exp_name: "{args.exp_name}"
  base_path: "{base_path}"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "kronos_custom_finetune_btc"
  description: "BTCUSDT 5m fine-tuning using downloaded klines"
  use_comet: false
  train_tokenizer: {bool_str(not args.skip_tokenizer)}
  train_basemodel: {bool_str(not args.skip_basemodel)}
  skip_existing: {bool_str(args.skip_existing)}

device:
  use_cuda: {bool_str(use_cuda)}
  device_id: 0
"""


def write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def model_dirs(base_path: Path, exp_name: str) -> Dict[str, Path]:
    exp_dir = base_path / exp_name
    return {
        "tokenizer_best": exp_dir / "tokenizer" / "best_model",
        "basemodel_best": exp_dir / "basemodel" / "best_model",
    }


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_positive_int_grid(raw: str, default_values: List[int], field_name: str) -> List[int]:
    if raw.strip():
        values: List[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"{field_name} must contain integers, got: {token!r}") from exc
            if value <= 0:
                raise ValueError(f"{field_name} values must be > 0, got: {value}")
            values.append(value)
        if not values:
            raise ValueError(f"{field_name} resolved to an empty candidate list")
    else:
        values = [int(v) for v in default_values if int(v) > 0]
        if not values:
            raise ValueError(f"{field_name} has no valid defaults")
    return sorted(set(values))


def default_lookback_grid(base_lookback: int) -> List[int]:
    half = max(16, base_lookback // 2)
    one = max(16, base_lookback)
    one_half = max(16, base_lookback + base_lookback // 2)
    two = max(16, base_lookback * 2)
    return [half, one, one_half, two]


def default_block_len_grid(base_block_len: int) -> List[int]:
    quarter = max(1, base_block_len // 4)
    half = max(1, base_block_len // 2)
    one = max(1, base_block_len)
    two = max(1, base_block_len * 2)
    return [quarter, half, one, two]


def build_eval_cmd(
    train_context_path: Path,
    eval_path: Path,
    model_id: str,
    tokenizer_id: str,
    lookback: int,
    block_len: int,
    feedback_source: str,
    device: str,
    output_csv: Path,
    output_json: Path,
) -> List[str]:
    return [
        sys.executable,
        "eval_consecutive_rollout.py",
        "--train-context-path",
        str(train_context_path),
        "--eval-path",
        str(eval_path),
        "--model-id",
        model_id,
        "--tokenizer-id",
        tokenizer_id,
        "--lookback",
        str(int(lookback)),
        "--block-len",
        str(int(block_len)),
        "--feedback-source",
        feedback_source,
        "--device",
        device,
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
    ]


def run_eval_and_get_metric(
    *,
    finetune_dir: Path,
    env: dict[str, str],
    train_context_path: Path,
    eval_path: Path,
    model_id: str,
    tokenizer_id: str,
    lookback: int,
    block_len: int,
    feedback_source: str,
    device: str,
    output_csv: Path,
    output_json: Path,
    metric_key: str,
) -> Tuple[float, dict]:
    run_cmd(
        build_eval_cmd(
            train_context_path=train_context_path,
            eval_path=eval_path,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            lookback=lookback,
            block_len=block_len,
            feedback_source=feedback_source,
            device=device,
            output_csv=output_csv,
            output_json=output_json,
        ),
        finetune_dir,
        env,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if metric_key not in payload:
        raise KeyError(f"Metric key {metric_key!r} not found in {output_json}")
    score = float(payload[metric_key])
    return score, payload


def auto_tune_eval_hparams(
    *,
    finetune_dir: Path,
    env: dict[str, str],
    train_context_path: Path,
    eval_path: Path,
    model_id: str,
    tokenizer_id: str,
    feedback_source: str,
    eval_device: str,
    eval_out_dir: Path,
    metric_key: str,
    lookback_candidates: List[int],
    block_len_candidates: List[int],
    max_trials: int,
) -> Tuple[int, int, Path]:
    trial_records = []
    successful_trials = []
    trial_count = 0

    for lookback in lookback_candidates:
        for block_len in block_len_candidates:
            if max_trials > 0 and trial_count >= max_trials:
                break
            trial_count += 1

            trial_tag = f"autotune_lb{lookback}_bl{block_len}"
            trial_csv = eval_out_dir / f"{trial_tag}.csv"
            trial_json = eval_out_dir / f"{trial_tag}.json"

            print(
                f"[auto-tune] trial {trial_count}: "
                f"lookback={lookback}, block_len={block_len}, metric={metric_key}"
            )
            try:
                score, payload = run_eval_and_get_metric(
                    finetune_dir=finetune_dir,
                    env=env,
                    train_context_path=train_context_path,
                    eval_path=eval_path,
                    model_id=model_id,
                    tokenizer_id=tokenizer_id,
                    lookback=lookback,
                    block_len=block_len,
                    feedback_source=feedback_source,
                    device=eval_device,
                    output_csv=trial_csv,
                    output_json=trial_json,
                    metric_key=metric_key,
                )
            except Exception as exc:
                print(f"[auto-tune] failed for lookback={lookback}, block_len={block_len}: {exc}")
                trial_records.append(
                    {
                        "lookback": int(lookback),
                        "block_len": int(block_len),
                        "status": "failed",
                        "error": str(exc),
                        "metric_key": metric_key,
                        "metric_value": None,
                        "output_csv": str(trial_csv),
                        "output_json": str(trial_json),
                    }
                )
                continue

            print(f"[auto-tune] score={score:.6f}")
            trial = {
                "lookback": int(lookback),
                "block_len": int(block_len),
                "status": "ok",
                "metric_key": metric_key,
                "metric_value": float(score),
                "rows": int(payload.get("rows", 0)),
                "range_pred_start": payload.get("range_pred_start"),
                "range_pred_end": payload.get("range_pred_end"),
                "output_csv": str(trial_csv),
                "output_json": str(trial_json),
            }
            trial_records.append(trial)
            successful_trials.append(trial)

        if max_trials > 0 and trial_count >= max_trials:
            break

    if not successful_trials:
        raise RuntimeError("Auto-tune found no successful lookback/block_len trials.")

    best = max(successful_trials, key=lambda t: (float(t["metric_value"]), -int(t["block_len"]), int(t["lookback"])))
    summary = {
        "metric_key": metric_key,
        "eval_path": str(eval_path),
        "train_context_path": str(train_context_path),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "feedback_source": feedback_source,
        "device": eval_device,
        "lookback_candidates": [int(v) for v in lookback_candidates],
        "block_len_candidates": [int(v) for v in block_len_candidates],
        "max_trials": int(max_trials),
        "trials_evaluated": int(trial_count),
        "successful_trials": int(len(successful_trials)),
        "best": best,
        "trials": trial_records,
    }
    summary_path = eval_out_dir / "autotune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[auto-tune] winner lookback={best['lookback']}, "
        f"block_len={best['block_len']}, {metric_key}={best['metric_value']:.6f}"
    )
    print(f"[auto-tune] summary: {summary_path}")
    return int(best["lookback"]), int(best["block_len"]), summary_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    finetune_dir = repo_root / "finetune_csv"

    data_path = Path(args.data_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    base_path = Path(args.base_path).expanduser().resolve()

    print("Step 1/4 - Validate BTC kline CSV")
    row_count, first_ts, last_ts = inspect_csv(data_path)
    min_required = args.lookback_window + args.predict_window + 1
    if row_count < min_required:
        raise ValueError(
            f"Not enough rows for windows (need >= {min_required}, got {row_count}). "
            "Increase history window."
        )
    print(f"Rows: {row_count}, Range: {first_ts} -> {last_ts}")

    print("Step 2/4 - Generate fine-tune config")
    config_text = build_config_text(args, data_path, base_path)
    write_config(config_path, config_text)
    print(f"Config saved: {config_path}")

    if args.prepare_only:
        print("prepare-only enabled. Exiting before training.")
        return

    dirs = model_dirs(base_path, args.exp_name)
    skip_tokenizer = args.skip_tokenizer
    skip_basemodel = args.skip_basemodel
    if args.skip_existing:
        if dirs["tokenizer_best"].exists():
            skip_tokenizer = True
            print(f"Tokenizer exists, skipping: {dirs['tokenizer_best']}")
        if dirs["basemodel_best"].exists():
            skip_basemodel = True
            print(f"Basemodel exists, skipping: {dirs['basemodel_best']}")

    env = os.environ.copy()
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    rel_config = os.path.relpath(config_path, finetune_dir)

    if args.mode == "sequential":
        print("Step 3/4 - Run sequential trainer")
        cmd = [sys.executable, "train_sequential.py", "--config", rel_config]
        if skip_tokenizer:
            cmd.append("--skip-tokenizer")
        if skip_basemodel:
            cmd.append("--skip-basemodel")
        if args.skip_existing:
            cmd.append("--skip-existing")
        run_cmd(cmd, finetune_dir, env)
    else:
        print("Step 3/4 - Fine-tune tokenizer")
        if skip_tokenizer:
            print("Tokenizer step skipped")
        else:
            run_cmd([sys.executable, "finetune_tokenizer.py", "--config", rel_config], finetune_dir, env)

        print("Step 4/4 - Fine-tune predictor (basemodel)")
        if skip_basemodel:
            print("Basemodel step skipped")
        else:
            run_cmd([sys.executable, "finetune_base_model.py", "--config", rel_config], finetune_dir, env)

    if args.run_consecutive_eval and args.eval_path:
        eval_path = Path(args.eval_path).expanduser().resolve()
        if not eval_path.exists():
            raise FileNotFoundError(f"Eval CSV not found: {eval_path}")

        model_id = str(dirs["basemodel_best"]) if dirs["basemodel_best"].exists() else args.predictor_pretrained
        tokenizer_id = str(dirs["tokenizer_best"]) if dirs["tokenizer_best"].exists() else args.tokenizer_pretrained

        eval_out_dir = base_path / args.exp_name / "eval"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.eval_output_prefix or f"consecutive_{args.eval_feedback_source}"
        eval_csv = eval_out_dir / f"{prefix}.csv"
        eval_json = eval_out_dir / f"{prefix}.json"
        eval_lookback = args.eval_lookback if args.eval_lookback > 0 else args.lookback_window
        eval_block_len = args.eval_block_len if args.eval_block_len > 0 else args.predict_window
        eval_device = "cpu" if args.force_cpu else "cuda:0"
        autotune_summary_path: Path | None = None

        if args.auto_tune_eval_hparams:
            lookback_candidates = parse_positive_int_grid(
                args.auto_tune_lookbacks,
                default_lookback_grid(eval_lookback),
                "--auto-tune-lookbacks",
            )
            block_len_candidates = parse_positive_int_grid(
                args.auto_tune_block_lens,
                default_block_len_grid(eval_block_len),
                "--auto-tune-block-lens",
            )
            print("Step 5/6 - Auto-tune eval lookback/block_len")
            eval_lookback, eval_block_len, autotune_summary_path = auto_tune_eval_hparams(
                finetune_dir=finetune_dir,
                env=env,
                train_context_path=data_path,
                eval_path=eval_path,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                feedback_source=args.eval_feedback_source,
                eval_device=eval_device,
                eval_out_dir=eval_out_dir,
                metric_key=args.auto_tune_metric,
                lookback_candidates=lookback_candidates,
                block_len_candidates=block_len_candidates,
                max_trials=args.auto_tune_max_trials,
            )
        eval_step_label = "Step 6/6" if args.auto_tune_eval_hparams else "Step 5/5"
        print(f"{eval_step_label} - Consecutive rollout evaluation")
        run_cmd(
            build_eval_cmd(
                train_context_path=data_path,
                eval_path=eval_path,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                lookback=eval_lookback,
                block_len=eval_block_len,
                feedback_source=args.eval_feedback_source,
                device=eval_device,
                output_csv=eval_csv,
                output_json=eval_json,
            ),
            finetune_dir,
            env,
        )
        print(f"Consecutive eval params: lookback={eval_lookback}, block_len={eval_block_len}")
        print(f"Consecutive eval JSON: {eval_json}")
        if autotune_summary_path is not None:
            print(f"Auto-tune summary JSON: {autotune_summary_path}")

    print("All requested steps completed.")


if __name__ == "__main__":
    main()
