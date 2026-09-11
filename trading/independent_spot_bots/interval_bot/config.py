from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .types import ExecutionEnvironment, PriceReference, decimal_from


LIVE_REST_BASE_URL = "https://api.binance.com"
TESTNET_REST_BASE_URL = "https://testnet.binance.vision"
LIVE_WS_BASE_URL = "wss://stream.binance.com:9443/ws"
TESTNET_WS_BASE_URL = "wss://stream.testnet.binance.vision/ws"


@dataclass(slots=True)
class BotConfig:
    config_path: Path
    bot_name: str
    client_order_id_prefix: str
    symbol: str
    timeframe: str
    lookback: int
    predlen: int
    model_type: str
    model_path: Path
    tokenizer_path: Path | None
    python_callable: str | None
    model_device: str
    model_clip: float
    model_direction_mode: str
    model_stabilize_output: bool
    model_stability_context_path: Path | None
    environment: ExecutionEnvironment
    dry_run: bool
    use_websocket_close_detection: bool
    polling_fallback_enabled: bool
    polling_interval_seconds: int
    quote_allocation_pct: Decimal
    base_allocation_pct: Decimal
    price_reference: PriceReference
    limit_price_offset_bps: Decimal
    prefer_maker_pricing: bool
    order_timeout_seconds: int
    cancel_replace_enabled: bool
    max_cancel_replace_attempts: int
    cancel_replace_interval_seconds: int
    order_status_poll_seconds: int
    log_file_path: Path
    state_file_path: Path
    api_key_env: str
    api_secret_env: str
    live_trading_env: str
    simulated_quote_balance: Decimal
    simulated_base_balance: Decimal
    use_official_sdk_public_endpoints: bool
    recv_window_ms: int
    request_timeout_seconds: int
    time_sync_interval_seconds: int
    rest_base_url: str
    ws_base_url: str

    @property
    def live_trading_enabled(self) -> bool:
        return os.getenv(self.live_trading_env, "false").strip().lower() == "true"


def load_env_file(env_path: Path | None) -> None:
    if env_path is None or not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_optional_path(base_dir: Path, raw_value: object | None) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return _resolve_path(base_dir, text)


def _bool(section: dict, key: str, default: bool) -> bool:
    return bool(section.get(key, default))


def _int(section: dict, key: str, default: int) -> int:
    return int(section.get(key, default))


def _decimal(section: dict, key: str, default: str) -> Decimal:
    return decimal_from(section.get(key, default))


def load_bot_config(config_path: Path, env_path: Path | None = None) -> BotConfig:
    config_path = config_path.expanduser().resolve()
    load_env_file(env_path)

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    base_dir = config_path.parent
    bot = raw.get("bot", {})
    runtime = raw.get("runtime", {})
    execution = raw.get("execution", {})
    credentials = raw.get("credentials", {})
    binance = raw.get("binance", {})

    environment = ExecutionEnvironment(runtime.get("environment", "testnet"))
    rest_base_url = str(
        binance.get(
            "rest_base_url",
            TESTNET_REST_BASE_URL if environment == ExecutionEnvironment.TESTNET else LIVE_REST_BASE_URL,
        )
    )
    ws_base_url = str(
        binance.get(
            "ws_base_url",
            TESTNET_WS_BASE_URL if environment == ExecutionEnvironment.TESTNET else LIVE_WS_BASE_URL,
        )
    )

    config = BotConfig(
        config_path=config_path,
        bot_name=str(bot["name"]),
        client_order_id_prefix=str(bot.get("client_order_id_prefix", bot["name"]))[:12],
        symbol=str(bot["symbol"]).upper(),
        timeframe=str(bot["timeframe"]),
        lookback=int(bot["lookback"]),
        predlen=int(bot["predlen"]),
        model_type=str(bot.get("model_type", "stub")).lower(),
        model_path=_resolve_path(base_dir, str(bot.get("model_path", "./models/stub.model"))),
        tokenizer_path=_resolve_optional_path(base_dir, bot.get("tokenizer_path")),
        python_callable=bot.get("python_callable"),
        model_device=str(bot.get("device", "cpu")),
        model_clip=float(bot.get("clip", 5.0)),
        model_direction_mode=str(bot.get("direction_mode", "close_to_close")),
        model_stabilize_output=_bool(bot, "stabilize_output", True),
        model_stability_context_path=_resolve_optional_path(base_dir, bot.get("stability_context_path")),
        environment=environment,
        dry_run=_bool(runtime, "dry_run", True),
        use_websocket_close_detection=_bool(runtime, "use_websocket_close_detection", True),
        polling_fallback_enabled=_bool(runtime, "polling_fallback_enabled", True),
        polling_interval_seconds=_int(runtime, "polling_interval_seconds", 15),
        quote_allocation_pct=_decimal(execution, "quote_allocation_pct", "0.99"),
        base_allocation_pct=_decimal(execution, "base_allocation_pct", "1.00"),
        price_reference=PriceReference(execution.get("price_reference", "book")),
        limit_price_offset_bps=_decimal(execution, "limit_price_offset_bps", "0"),
        prefer_maker_pricing=_bool(execution, "prefer_maker_pricing", True),
        order_timeout_seconds=_int(execution, "order_timeout_seconds", 45),
        cancel_replace_enabled=_bool(execution, "cancel_replace_enabled", True),
        max_cancel_replace_attempts=_int(execution, "max_cancel_replace_attempts", 1),
        cancel_replace_interval_seconds=_int(execution, "cancel_replace_interval_seconds", 5),
        order_status_poll_seconds=_int(execution, "order_status_poll_seconds", 2),
        log_file_path=_resolve_path(base_dir, str(runtime["log_file_path"])),
        state_file_path=_resolve_path(base_dir, str(runtime["state_file_path"])),
        api_key_env=str(credentials.get("api_key_env", "BINANCE_API_KEY")),
        api_secret_env=str(credentials.get("api_secret_env", "BINANCE_API_SECRET")),
        live_trading_env=str(credentials.get("live_trading_env", "LIVE_TRADING")),
        simulated_quote_balance=_decimal(runtime, "simulated_quote_balance", "1000"),
        simulated_base_balance=_decimal(runtime, "simulated_base_balance", "0"),
        use_official_sdk_public_endpoints=_bool(binance, "use_official_sdk_public_endpoints", True),
        recv_window_ms=_int(binance, "recv_window_ms", 5000),
        request_timeout_seconds=_int(binance, "request_timeout_seconds", 10),
        time_sync_interval_seconds=_int(binance, "time_sync_interval_seconds", 1800),
        rest_base_url=rest_base_url,
        ws_base_url=ws_base_url,
    )
    _validate_config(config)
    return config


def _validate_config(config: BotConfig) -> None:
    if config.predlen not in {1, 16}:
        raise ValueError("predlen must be 1 or 16")
    if config.lookback <= 0:
        raise ValueError("lookback must be positive")
    if config.model_clip <= 0:
        raise ValueError("clip must be positive")
    if not Decimal("0") < config.quote_allocation_pct <= Decimal("1"):
        raise ValueError("quote_allocation_pct must be in (0, 1]")
    if not Decimal("0") < config.base_allocation_pct <= Decimal("1"):
        raise ValueError("base_allocation_pct must be in (0, 1]")
    if config.order_timeout_seconds <= 0:
        raise ValueError("order_timeout_seconds must be positive")
    if config.cancel_replace_interval_seconds <= 0:
        raise ValueError("cancel_replace_interval_seconds must be positive")
    if config.polling_interval_seconds <= 0:
        raise ValueError("polling_interval_seconds must be positive")
    if not config.use_websocket_close_detection and not config.polling_fallback_enabled:
        raise ValueError("close detection requires websocket or polling fallback to be enabled")
    if config.model_type == "kronos_consecutive" and config.tokenizer_path is None:
        raise ValueError("tokenizer_path is required when model_type='kronos_consecutive'")
    if config.environment == ExecutionEnvironment.LIVE and not config.dry_run and not config.live_trading_enabled:
        raise ValueError(
            f"Refusing live order mode without {config.live_trading_env}=true in the environment"
        )
