#!/usr/bin/env python3
"""
Run fine-tuning jobs in a continuous round-robin cycle.

When one job exits (success or failure), the runner immediately starts the next
job in the configured list. This keeps training active without manual relaunch.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Job:
    name: str
    command: str


DEFAULT_JOBS: List[Job] = [
    Job(
        name="btc_1d_2018_2025",
        command="python3 train_sequential.py --config configs/config_btcusdt_1d_2018_2025.yaml",
    ),
    Job(
        name="btc_1h_2018_2025",
        command="python3 train_sequential.py --config configs/config_btcusdt_1h_2018_2025.yaml",
    ),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_compact(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def ts_human(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_job(raw: str) -> Job:
    if "::" not in raw:
        raise ValueError(
            f"Invalid --job '{raw}'. Use format: name::command"
        )
    name, command = raw.split("::", 1)
    name = name.strip()
    command = command.strip()
    if not name:
        raise ValueError(f"Invalid --job '{raw}': empty name")
    if not command:
        raise ValueError(f"Invalid --job '{raw}': empty command")
    return Job(name=name, command=command)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    finetune_dir = repo_root / "finetune_csv"
    logs_dir = finetune_dir / "logs"

    parser = argparse.ArgumentParser(
        description="Continuously rotate through fine-tune jobs."
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help=(
            "Job definition in format name::command. "
            "Repeat --job to define multiple jobs. "
            "If omitted, defaults to 1d then 1h sequential training."
        ),
    )
    parser.add_argument(
        "--finetune-dir",
        type=str,
        default=str(finetune_dir),
        help="Working directory where each job command is executed.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(logs_dir / "auto_finetune_cycle"),
        help="Directory for per-run job logs.",
    )
    parser.add_argument(
        "--runner-log",
        type=str,
        default=str(logs_dir / "auto_finetune_cycle.log"),
        help="Runner status log path.",
    )
    parser.add_argument(
        "--state-jsonl",
        type=str,
        default=str(logs_dir / "auto_finetune_cycle_state.jsonl"),
        help="JSONL state file with one record per finished run.",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=str(finetune_dir / "auto_finetune_cycle.pid"),
        help="PID file for this runner.",
    )
    parser.add_argument(
        "--current-job-file",
        type=str,
        default=str(finetune_dir / "auto_finetune_cycle.current"),
        help="Current job metadata file.",
    )
    parser.add_argument(
        "--sleep-after-success-sec",
        type=int,
        default=3,
        help="Sleep before launching the next job after a successful run.",
    )
    parser.add_argument(
        "--sleep-after-failure-sec",
        type=int,
        default=10,
        help="Sleep before launching the next job after a failed run.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Total full cycles to run (0 means run forever).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Exit runner when any job fails.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def log(msg: str, runner_log: Path) -> None:
    line = f"[{ts_human(utc_now())}] {msg}"
    print(line, flush=True)
    append_text(runner_log, line + "\n")


def append_state(path: Path, payload: dict) -> None:
    append_text(path, json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()

    jobs = [parse_job(raw) for raw in args.job] if args.job else DEFAULT_JOBS
    if not jobs:
        raise ValueError("No jobs configured.")

    finetune_dir = Path(args.finetune_dir).expanduser().resolve()
    if not finetune_dir.exists():
        raise FileNotFoundError(f"finetune dir not found: {finetune_dir}")

    log_dir = Path(args.log_dir).expanduser().resolve()
    runner_log = Path(args.runner_log).expanduser().resolve()
    state_jsonl = Path(args.state_jsonl).expanduser().resolve()
    pid_file = Path(args.pid_file).expanduser().resolve()
    current_job_file = Path(args.current_job_file).expanduser().resolve()

    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log.parent.mkdir(parents=True, exist_ok=True)
    state_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    current_job_file.parent.mkdir(parents=True, exist_ok=True)

    write_text(pid_file, str(os.getpid()) + "\n")
    stop = False
    active_proc: subprocess.Popen | None = None

    def handle_stop(signum: int, _frame) -> None:
        nonlocal stop
        stop = True
        log(f"received signal {signum}; stopping after current action", runner_log)
        if active_proc is not None and active_proc.poll() is None:
            try:
                active_proc.terminate()
                log(f"sent terminate to child pid={active_proc.pid}", runner_log)
            except Exception as exc:  # pragma: no cover
                log(f"failed to terminate child: {exc}", runner_log)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    log(
        "auto cycle started: "
        + ", ".join(f"{j.name} -> {j.command}" for j in jobs),
        runner_log,
    )

    cycle_idx = 0
    run_idx = 0

    try:
        while not stop:
            cycle_idx += 1
            if args.max_cycles > 0 and cycle_idx > args.max_cycles:
                log(f"max cycles reached ({args.max_cycles}); exiting", runner_log)
                break

            for job in jobs:
                if stop:
                    break

                run_idx += 1
                start_dt = utc_now()
                run_tag = f"{ts_compact(start_dt)}_{run_idx:06d}_{job.name}"
                run_log_path = log_dir / f"{run_tag}.log"

                write_text(
                    current_job_file,
                    json.dumps(
                        {
                            "runner_pid": os.getpid(),
                            "run_index": run_idx,
                            "cycle_index": cycle_idx,
                            "job_name": job.name,
                            "command": job.command,
                            "start_utc": ts_human(start_dt),
                            "run_log": str(run_log_path),
                        },
                        ensure_ascii=True,
                        indent=2,
                    )
                    + "\n",
                )

                log(
                    f"starting run={run_idx} cycle={cycle_idx} job={job.name} log={run_log_path}",
                    runner_log,
                )

                with run_log_path.open("w", encoding="utf-8") as run_log:
                    run_log.write(f"[{ts_human(start_dt)}] command: {job.command}\n")
                    run_log.flush()

                    active_proc = subprocess.Popen(
                        ["/bin/bash", "-lc", job.command],
                        cwd=str(finetune_dir),
                        stdout=run_log,
                        stderr=subprocess.STDOUT,
                    )
                    rc = active_proc.wait()
                    end_dt = utc_now()

                duration_sec = int((end_dt - start_dt).total_seconds())
                status = "ok" if rc == 0 else "failed"
                log(
                    f"finished run={run_idx} cycle={cycle_idx} job={job.name} status={status} rc={rc} duration={duration_sec}s",
                    runner_log,
                )

                append_state(
                    state_jsonl,
                    {
                        "run_index": run_idx,
                        "cycle_index": cycle_idx,
                        "job_name": job.name,
                        "command": job.command,
                        "start_utc": ts_human(start_dt),
                        "end_utc": ts_human(end_dt),
                        "duration_sec": duration_sec,
                        "rc": rc,
                        "status": status,
                        "run_log": str(run_log_path),
                    },
                )

                active_proc = None

                if rc != 0 and args.stop_on_failure:
                    log("stop-on-failure enabled; exiting", runner_log)
                    stop = True
                    break

                sleep_sec = (
                    max(0, args.sleep_after_success_sec)
                    if rc == 0
                    else max(0, args.sleep_after_failure_sec)
                )
                if sleep_sec > 0 and not stop:
                    log(f"sleeping {sleep_sec}s before next job", runner_log)
                    end_sleep = time.time() + sleep_sec
                    while not stop and time.time() < end_sleep:
                        time.sleep(1)

    finally:
        if active_proc is not None and active_proc.poll() is None:
            try:
                active_proc.terminate()
            except Exception:
                pass
        write_text(current_job_file, "{}\n")
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        log("auto cycle stopped", runner_log)


if __name__ == "__main__":
    main()
