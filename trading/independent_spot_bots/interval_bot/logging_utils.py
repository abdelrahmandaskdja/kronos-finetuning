from __future__ import annotations

import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .types import current_time_offset_ms


class ExchangeAlignedFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        aligned_created = float(record.created) + (current_time_offset_ms() / 1000.0)
        dt = datetime.fromtimestamp(aligned_created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def setup_logging(bot_name: str, log_file_path: Path) -> logging.Logger:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"interval_bot.{bot_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = ExchangeAlignedFormatter(
        "%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
        "%Y-%m-%dT%H:%M:%S%z",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
