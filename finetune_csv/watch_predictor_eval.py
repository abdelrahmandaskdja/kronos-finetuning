#!/usr/bin/env python3
"""Watch a local best-model checkpoint and re-run directional evaluation on update."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.evaluate_directional_accuracy import (  # noqa: E402
    evaluate_directional_accuracy,
    load_kline_csv,
    set_seed,
)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_signature(path: Path) -> Optional[Tuple[int, int]]:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def resolve_checkpoint_file(model_id: str) -> Optional[Path]:
    path = Path(model_id)
    if not path.exists():
        return None
    if path.is_file():
        return path

    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.exists():
            return candidate

    # Fallback: expected checkpoint filename for local save_pretrained runs.
    return path / "model.safetensors"


def parse_args() -> argparse.Namespace:
    default_model = REPO_ROOT / "finetune_csv" / "finetuned" / "BTCUSDT_kline_5min_2018_2025" / "basemodel" / "best_model"
    default_tokenizer = REPO_ROOT / "finetune_csv" / "finetuned" / "BTCUSDT_kline_5min_2018_2025" / "tokenizer" / "best_model"
    default_data = REPO_ROOT / "finetune_csv" / "data" / "BTCUSDT_kline_5min_2026_to_now.csv"
    default_output_dir = REPO_ROOT / "finetune_csv" / "eval_watch_predictor_2026"
    default_state_file = REPO_ROOT / "finetune_csv" / "eval_watch_predictor_2026_state.json"

    parser = argparse.ArgumentParser(
        description="Watch local best checkpoint and run directional evaluation whenever it updates."
    )
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--model-id", type=str, default=str(default_model))
    parser.add_argument("--tokenizer-id", type=str, default=str(default_tokenizer))
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--num-windows", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--settle-seconds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--state-file", type=Path, default=default_state_file)
    parser.add_argument("--log-file", type=Path, default=None)

    parser.add_argument("--once", action="store_true", help="Run one evaluation and exit.")
    parser.add_argument(
        "--eval-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When watching, evaluate at startup if current checkpoint signature differs from state.",
    )
    return parser.parse_args()


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_file: Path, checkpoint_signature: Optional[Tuple[int, int]], last_output: Optional[Path]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": utc_now(),
        "checkpoint_signature": list(checkpoint_signature) if checkpoint_signature else None,
        "last_output_json": str(last_output) if last_output else None,
    }
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_logger(log_file: Optional[Path]):
    def _log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    return _log


def run_evaluation(args: argparse.Namespace, df, checkpoint_file: Optional[Path], log) -> Path:
    set_seed(args.seed)
    start = time.time()

    log(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer_id)
    log(f"Loading model: {args.model_id}")
    model = Kronos.from_pretrained(args.model_id)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_context=args.max_context,
        clip=args.clip,
    )

    eval_args = argparse.Namespace(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        data_path=args.data_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        num_windows=args.num_windows,
        stride=args.stride,
        batch_size=args.batch_size,
        max_context=args.max_context,
        clip=args.clip,
        T=args.T,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=args.sample_count,
        device=args.device,
    )
    metrics = evaluate_directional_accuracy(df=df, predictor=predictor, args=eval_args)
    metrics["eval_timestamp_utc"] = utc_now()
    metrics["elapsed_seconds"] = time.time() - start

    if checkpoint_file is not None:
        sig = get_signature(checkpoint_file)
        metrics["checkpoint_file"] = str(checkpoint_file)
        metrics["checkpoint_signature"] = {
            "mtime_ns": sig[0] if sig else None,
            "size": sig[1] if sig else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_json = args.output_dir / f"eval_{stamp}.json"
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history_jsonl = args.output_dir / "history.jsonl"
    with history_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    log(
        "Evaluation complete: "
        f"acc_close_vs_last={metrics['directional_accuracy_close_vs_last_close']:.6f}, "
        f"baseline={metrics['majority_baseline_accuracy']:.6f}, "
        f"pred_up_ratio={metrics['predicted_up_ratio']:.6f}, "
        f"output={output_json}"
    )
    return output_json


def main() -> None:
    args = parse_args()
    log = make_logger(args.log_file)

    log(f"Loading data once from: {args.data_path}")
    df = load_kline_csv(args.data_path)
    log(f"Loaded rows: {len(df)}")

    checkpoint_file = resolve_checkpoint_file(args.model_id)
    if args.once:
        output_json = run_evaluation(args=args, df=df, checkpoint_file=checkpoint_file, log=log)
        save_state(args.state_file, get_signature(checkpoint_file) if checkpoint_file else None, output_json)
        return

    state = load_state(args.state_file)
    last_signature_raw = state.get("checkpoint_signature")
    last_signature = tuple(last_signature_raw) if isinstance(last_signature_raw, list) else None

    log(f"Watching model id: {args.model_id}")
    if checkpoint_file is not None:
        log(f"Monitoring checkpoint file: {checkpoint_file}")
    else:
        log("Model id does not map to a local file yet. Waiting for local checkpoint path to appear.")

    if args.eval_on_start:
        current_sig = get_signature(checkpoint_file) if checkpoint_file else None
        if current_sig is not None and current_sig != last_signature:
            log("Startup evaluation triggered (checkpoint differs from state).")
            output_json = run_evaluation(args=args, df=df, checkpoint_file=checkpoint_file, log=log)
            last_signature = current_sig
            save_state(args.state_file, last_signature, output_json)
        else:
            log("Startup evaluation skipped (no new checkpoint signature).")

    while True:
        try:
            checkpoint_file = resolve_checkpoint_file(args.model_id)
            current_sig = get_signature(checkpoint_file) if checkpoint_file else None

            if current_sig is None:
                log("Checkpoint file not found yet. Waiting...")
            elif current_sig != last_signature:
                log(f"Detected checkpoint update: {last_signature} -> {current_sig}")
                time.sleep(max(1, args.settle_seconds))
                settled_sig = get_signature(checkpoint_file)
                if settled_sig != current_sig:
                    log(
                        "Checkpoint still changing after settle wait; "
                        f"will retry next poll (now {settled_sig})."
                    )
                else:
                    output_json = run_evaluation(args=args, df=df, checkpoint_file=checkpoint_file, log=log)
                    last_signature = settled_sig
                    save_state(args.state_file, last_signature, output_json)

            time.sleep(max(5, args.poll_seconds))
        except KeyboardInterrupt:
            log("Stopped by user.")
            return
        except Exception as exc:
            log(f"Watcher error: {exc}. Retrying after poll interval.")
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()

