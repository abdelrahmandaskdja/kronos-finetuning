from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from .config import load_bot_config
from .engine import SpotIntervalBot
from .logging_utils import setup_logging


async def _async_run(config_path: Path, env_path: Path | None, once: bool) -> int:
    config = load_bot_config(config_path, env_path=env_path)
    logger = setup_logging(config.bot_name, config.log_file_path)
    bot = SpotIntervalBot(config, logger)
    loop = asyncio.get_running_loop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, bot.request_stop)
        except NotImplementedError:
            pass

    await bot.run_forever(process_once=once)
    return 0


def run_bot_from_config(default_config_path: Path) -> int:
    parser = argparse.ArgumentParser(description="Independent Binance Spot interval bot runner")
    parser.add_argument("--config", default=str(default_config_path), help="Path to the bot TOML config file")
    parser.add_argument(
        "--env-file",
        default=str(default_config_path.parent.parent / ".env"),
        help="Optional .env file for secrets and LIVE_TRADING",
    )
    parser.add_argument("--once", action="store_true", help="Process catch-up work once and exit")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve() if args.env_file else None
    return asyncio.run(_async_run(config_path=config_path, env_path=env_path, once=args.once))
