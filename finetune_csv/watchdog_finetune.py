#!/usr/bin/env python3
"""
Watchdog for Kronos finetuning jobs.

Monitors:
1) Training PID is alive.
2) Training log keeps advancing step lines: [Epoch X/Y, Step A/B]
3) Alerts when progress stalls beyond a configured threshold.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


STEP_RE = re.compile(r"\[Epoch\s+(\d+)/(\d+),\s*Step\s+(\d+)/(\d+)\]")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log_line(msg: str, log_path: Path) -> None:
    text = f"[{utc_now()}] {msg}"
    print(text, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def tail_lines(path: Path, n: int) -> list[str]:
    dq: deque[str] = deque(maxlen=max(1, n))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def latest_step_from_lines(lines: list[str]) -> Optional[Tuple[int, int, int, int, str]]:
    for line in reversed(lines):
        m = STEP_RE.search(line)
        if m:
            ep, ep_total, step, step_total = map(int, m.groups())
            return ep, ep_total, step, step_total, line
    return None


def resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_file:
        return Path(args.log_file).expanduser().resolve()
    p = Path(args.log_path_file).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"log-path file not found: {p}")
    log_str = p.read_text(encoding="utf-8").strip()
    if not log_str:
        raise ValueError(f"log-path file is empty: {p}")
    return Path(log_str).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watchdog for Kronos fine-tuning")
    parser.add_argument(
        "--pid-file",
        type=str,
        default="finetune_csv/finetune_2018_2025_train.pid",
        help="Path to training PID file",
    )
    parser.add_argument(
        "--log-path-file",
        type=str,
        default="finetune_csv/finetune_2018_2025.logpath",
        help="Path to file that stores the training log path",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Training log path (overrides --log-path-file)",
    )
    parser.add_argument(
        "--watchdog-log",
        type=str,
        default="finetune_csv/logs/finetune_watchdog_2018_2025.log",
        help="Where watchdog status lines are written",
    )
    parser.add_argument(
        "--alert-log",
        type=str,
        default="finetune_csv/logs/finetune_watchdog_alerts_2018_2025.log",
        help="Where alert lines are written",
    )
    parser.add_argument(
        "--check-interval-sec",
        type=int,
        default=60,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--stall-minutes",
        type=int,
        default=20,
        help="Alert when no step/log-size progress for this many minutes",
    )
    parser.add_argument(
        "--repeat-alert-minutes",
        type=int,
        default=15,
        help="While stalled, emit repeated alerts every N minutes",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=500,
        help="Number of trailing lines to scan for step progress",
    )
    parser.add_argument(
        "--exit-on-process-end",
        action="store_true",
        help="Exit watchdog once training process is no longer alive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pid_file = Path(args.pid_file).expanduser().resolve()
    train_log = resolve_log_path(args)
    watchdog_log = Path(args.watchdog_log).expanduser().resolve()
    alert_log = Path(args.alert_log).expanduser().resolve()

    stall_sec = max(1, args.stall_minutes) * 60
    repeat_alert_sec = max(1, args.repeat_alert_minutes) * 60
    interval = max(2, args.check_interval_sec)

    last_step_key: Optional[Tuple[int, int, int, int]] = None
    last_step_ts = time.time()
    last_size_ts = time.time()
    last_file_size = -1

    stalled = False
    last_alert_ts = 0.0
    stop = False

    def handle_sigterm(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    log_line(
        (
            f"watchdog started: pid_file={pid_file}, train_log={train_log}, "
            f"interval={interval}s, stall={args.stall_minutes}m"
        ),
        watchdog_log,
    )

    while not stop:
        now = time.time()
        pid = read_pid(pid_file)
        alive = bool(pid is not None and process_alive(pid))

        if train_log.exists():
            size = train_log.stat().st_size
            if size != last_file_size:
                last_file_size = size
                last_size_ts = now

            lines = tail_lines(train_log, args.tail_lines)
            latest = latest_step_from_lines(lines)
            if latest is not None:
                ep, ep_total, step, step_total, src_line = latest
                step_key = (ep, ep_total, step, step_total)
                if step_key != last_step_key:
                    last_step_key = step_key
                    last_step_ts = now
                    if stalled:
                        stalled = False
                        log_line(
                            f"recovered: progress resumed at Epoch {ep}/{ep_total} Step {step}/{step_total}",
                            watchdog_log,
                        )
                    else:
                        log_line(
                            f"progress: Epoch {ep}/{ep_total} Step {step}/{step_total}",
                            watchdog_log,
                        )
                else:
                    # keep a lightweight heartbeat every interval
                    log_line(
                        (
                            f"heartbeat: pid_alive={alive}, last_step=Epoch {ep}/{ep_total} "
                            f"Step {step}/{step_total}, no_step_change={(now - last_step_ts):.0f}s, "
                            f"line='{src_line}'"
                        ),
                        watchdog_log,
                    )
            else:
                log_line(
                    f"heartbeat: pid_alive={alive}, no step line yet, log_size={last_file_size}",
                    watchdog_log,
                )
        else:
            log_line(f"heartbeat: pid_alive={alive}, waiting for train log: {train_log}", watchdog_log)

        inactivity = now - max(last_step_ts, last_size_ts)
        should_alert = inactivity >= stall_sec
        if should_alert:
            can_repeat = (now - last_alert_ts) >= repeat_alert_sec
            if (not stalled) or can_repeat:
                stalled = True
                last_alert_ts = now
                msg = (
                    f"ALERT stalled: no progress for {inactivity/60:.1f}m "
                    f"(pid_alive={alive}, pid={pid}, log={train_log})"
                )
                log_line(msg, watchdog_log)
                log_line(msg, alert_log)

        if not alive:
            msg = f"process ended: pid={pid} is not alive"
            log_line(msg, watchdog_log)
            log_line(f"ALERT {msg}", alert_log)
            if args.exit_on_process_end:
                break

        # Sleep in small chunks to exit quickly on signal.
        end_ts = time.time() + interval
        while not stop and time.time() < end_ts:
            time.sleep(1)

    log_line("watchdog stopped", watchdog_log)


if __name__ == "__main__":
    main()

