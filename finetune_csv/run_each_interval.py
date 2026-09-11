#!/usr/bin/env python3
"""Run one command for each selected interval config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = REPO_ROOT / "finetune_csv"
DEFAULT_LOG_DIR = DEFAULT_WORKDIR / "logs" / "run_each_interval"
DEFAULT_SUMMARY_JSON = DEFAULT_WORKDIR / "logs" / "run_each_interval_summary.json"
DEFAULT_COMMAND_TEMPLATE = "python3 train_sequential.py --config {config}"
DEFAULT_INTERVAL_CONFIGS: "OrderedDict[str, Path]" = OrderedDict(
    [
        ("5min", Path("configs/config_btcusdt_5min_2018_2025.yaml")),
        ("1h", Path("configs/config_btcusdt_1h_2018_2025.yaml")),
        ("1d", Path("configs/config_btcusdt_1d_2018_2025.yaml")),
    ]
)


@dataclass(frozen=True)
class IntervalJob:
    interval: str
    config_path: Path
    display_config: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_compact(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def ts_human(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def sanitize_slug(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() else "_")
    return "".join(keep).strip("_") or "job"


def parse_interval_config(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --interval-config {raw!r}. Use format interval=path/to/config.yaml"
        )
    interval, path_text = raw.split("=", 1)
    interval = interval.strip()
    path_text = path_text.strip()
    if not interval:
        raise ValueError(f"Invalid --interval-config {raw!r}: empty interval")
    if not path_text:
        raise ValueError(f"Invalid --interval-config {raw!r}: empty path")
    return interval, Path(path_text)


def resolve_path(path: Path, workdir: Path) -> Path:
    return path if path.is_absolute() else (workdir / path).resolve()


def display_path(path: Path, workdir: Path) -> str:
    try:
        return os.path.relpath(path, workdir)
    except ValueError:
        return str(path)


def build_interval_map(args: argparse.Namespace) -> "OrderedDict[str, Path]":
    interval_map: "OrderedDict[str, Path]" = OrderedDict(DEFAULT_INTERVAL_CONFIGS)
    for raw in args.interval_config:
        interval, path = parse_interval_config(raw)
        interval_map[interval] = path
    return interval_map


def build_jobs(args: argparse.Namespace) -> list[IntervalJob]:
    workdir = Path(args.workdir).expanduser().resolve()
    interval_map = build_interval_map(args)
    selected_intervals = args.interval or list(interval_map.keys())
    jobs: list[IntervalJob] = []

    for interval in selected_intervals:
        if interval not in interval_map:
            available = ", ".join(interval_map.keys())
            raise KeyError(
                f"Unknown interval {interval!r}. Known intervals: {available}"
            )
        config_path = resolve_path(interval_map[interval], workdir)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config for interval {interval!r} not found: {config_path}"
            )
        jobs.append(
            IntervalJob(
                interval=interval,
                config_path=config_path,
                display_config=display_path(config_path, workdir),
            )
        )

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command once for each selected interval config."
    )
    parser.add_argument(
        "--interval",
        action="append",
        default=[],
        help=(
            "Interval to run. Repeat to control order. "
            "Default order is 5min, 1h, 1d."
        ),
    )
    parser.add_argument(
        "--interval-config",
        action="append",
        default=[],
        help=(
            "Override or add an interval config in the form interval=path/to/config.yaml. "
            "Repeat as needed."
        ),
    )
    parser.add_argument(
        "--command-template",
        type=str,
        default=DEFAULT_COMMAND_TEMPLATE,
        help=(
            "Command template to run for each interval. "
            "Available placeholders: {interval}, {config}, {config_name}, {config_stem}."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument appended after the command template. "
            "Repeat as needed. Use --extra-arg=--flag for option-like values."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(DEFAULT_WORKDIR),
        help="Working directory used when running each command.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help="Directory for per-interval run logs.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=str(DEFAULT_SUMMARY_JSON),
        help="JSON summary written after the runner finishes.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later intervals even if one interval fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir not found: {workdir}")

    jobs = build_jobs(args)
    log_dir = Path(args.log_dir).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    print(
        f"[{ts_human(started_at)}] running {len(jobs)} interval job(s) from {workdir}",
        flush=True,
    )

    runs: list[dict[str, object]] = []
    failures = 0

    for index, job in enumerate(jobs, start=1):
        run_started_at = utc_now()
        command_text = args.command_template.format(
            interval=job.interval,
            config=job.display_config,
            config_name=job.config_path.name,
            config_stem=job.config_path.stem,
        )
        cmd = shlex.split(command_text)
        if args.extra_arg:
            cmd.extend(args.extra_arg)
        log_name = f"{ts_compact(run_started_at)}_{index:02d}_{sanitize_slug(job.interval)}.log"
        log_path = log_dir / log_name

        print(
            f"[{ts_human(run_started_at)}] [{index}/{len(jobs)}] interval={job.interval} "
            f"config={job.display_config}",
            flush=True,
        )
        print(f"$ {' '.join(shlex.quote(token) for token in cmd)}", flush=True)
        print(f"log: {log_path}", flush=True)

        run_info: dict[str, object] = {
            "index": index,
            "interval": job.interval,
            "config": str(job.config_path),
            "display_config": job.display_config,
            "command": cmd,
            "log_path": str(log_path),
            "start_utc": ts_human(run_started_at),
            "dry_run": bool(args.dry_run),
        }

        if args.dry_run:
            run_info["returncode"] = 0
            run_info["status"] = "dry_run"
            run_info["end_utc"] = ts_human(utc_now())
            runs.append(run_info)
            continue

        with log_path.open("w", encoding="utf-8") as run_log:
            run_log.write(
                f"[{ts_human(run_started_at)}] interval={job.interval} "
                f"config={job.display_config}\n"
            )
            run_log.write(f"command: {' '.join(shlex.quote(token) for token in cmd)}\n\n")
            run_log.flush()
            result = subprocess.run(
                cmd,
                cwd=str(workdir),
                stdout=run_log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        end_time = utc_now()
        run_info["returncode"] = result.returncode
        run_info["status"] = "ok" if result.returncode == 0 else "failed"
        run_info["end_utc"] = ts_human(end_time)
        runs.append(run_info)

        print(
            f"[{ts_human(end_time)}] interval={job.interval} finished with return code {result.returncode}",
            flush=True,
        )

        if result.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                break

    finished_at = utc_now()
    payload = {
        "workdir": str(workdir),
        "started_at_utc": ts_human(started_at),
        "finished_at_utc": ts_human(finished_at),
        "dry_run": bool(args.dry_run),
        "intervals_requested": args.interval or list(build_interval_map(args).keys()),
        "command_template": args.command_template,
        "extra_args": args.extra_arg,
        "runs": runs,
        "failure_count": failures,
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"summary: {summary_json}", flush=True)

    if failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
