#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interval_bot.runner import run_bot_from_config


if __name__ == "__main__":
    raise SystemExit(run_bot_from_config(PROJECT_ROOT / "configs" / "bot_1d.toml"))
