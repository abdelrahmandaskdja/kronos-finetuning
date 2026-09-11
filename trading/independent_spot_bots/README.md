# Independent Binance Spot Interval Bots

This package implements four separate long-only Binance Spot bots, one process per interval:

- `15m` bot with `predlen=1`
- `1h` bot with `predlen=16`
- `4h` bot with `predlen=16`
- `1d` bot with `predlen=16`

Each bot has its own:

- entrypoint script
- TOML config
- API credential env vars
- state file
- rotating log file

The bots never combine intervals into one trading loop.

## Assumptions

- The unspecified three higher intervals are assumed to be `1h`, `4h`, and `1d`.
- The `15m` model lookback was not provided, so the example config defaults it to `512`.
- The official maintained Binance Spot Python SDK as of 2026-04-14 is `binance-sdk-spot`. This code pins that package and uses it for public REST calls when available, with raw REST/websocket fallback for operational resilience.
- Dry-run defaults to a simulated `1000` quote balance and `0` base balance if credentials are not present.
- For flattening, the engine sells the full free base balance, rounded down to Binance lot size rules. Small dust can remain because of exchange filters.

## Features

- Spot only. No futures, margin, or shorting.
- Limit orders only.
- Two-state regime model: `LONG` or `CASH`.
- Closed-candle trading only.
- Websocket kline close detection first, REST fallback second.
- For `predlen=16`, the engine stores one 16-step forecast and consumes exactly one stored step per closed candle.
- Forecast cycles persist across restarts in the state file.
- Exchange filter enforcement for tick size, step size, min quantity, and notional.
- Graceful shutdown, rotating logs, retry logic, and Binance server-time sync.

## Layout

```text
trading/independent_spot_bots/
├── .env.example
├── README.md
├── requirements.txt
├── configs/
│   ├── bot_15m.toml
│   ├── bot_1d.toml
│   ├── bot_1h.toml
│   └── bot_4h.toml
├── interval_bot/
│   ├── __init__.py
│   ├── config.py
│   ├── engine.py
│   ├── exchange.py
│   ├── logging_utils.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── runner.py
│   ├── state.py
│   └── types.py
├── run_15m_bot.py
├── run_1d_bot.py
├── run_1h_bot.py
├── run_4h_bot.py
└── tests/
    ├── test_forecast_cycle.py
    └── test_state.py
```

## Setup

1. Create a Python 3.11 virtual environment.
2. Install the bot dependencies.
3. Copy `.env.example` to `.env`.
4. Fill in the API key and secret for each bot you want to run.
5. Keep `LIVE_TRADING=false` unless you explicitly want live spot orders.

Example:

```bash
cd trading/independent_spot_bots
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running

The defaults are `environment = "testnet"` and `dry_run = true`.

Run the four bots in separate terminals, tmux panes, or services:

```bash
python run_15m_bot.py
python run_1h_bot.py
python run_4h_bot.py
python run_1d_bot.py
```

To process current catch-up work once and exit:

```bash
python run_15m_bot.py --once
```

To point a runner at a different config:

```bash
python run_1h_bot.py --config ./configs/bot_1h.toml
```

## Config Notes

Each bot config contains:

- `symbol`
- `timeframe`
- `lookback`
- `predlen`
- `model_path`
- `environment`
- `quote_allocation_pct`
- `limit_price_offset_bps`
- `order_timeout_seconds`
- `cancel_replace_enabled`
- `polling_fallback_enabled`
- `log_file_path`
- `state_file_path`

Useful switches:

- `dry_run = true` keeps everything simulated.
- `environment = "testnet"` uses Binance Spot testnet endpoints.
- `environment = "live"` plus `LIVE_TRADING=true` plus `dry_run = false` is required for live spot orders.
- `price_reference = "book"` joins the best bid for buys or best ask for sells.
- `price_reference = "close_offset"` uses the latest closed candle close with a side-aware offset in bps.

## Model Adapter Options

Supported `model_type` values:

- `stub`
- `python_function`
- `joblib`
- `pytorch`
- `tensorflow`

The stub is active by default and keeps the trading engine runnable without a real trained model.

For a custom Python inference hook:

```toml
[bot]
model_type = "python_function"
python_callable = "my_models.live_inference:predict"
model_path = "../models/my_model_bundle"
```

The callable receives:

- `payload["features"]`: `numpy.ndarray` of shape `(lookback, feature_count)`
- `payload["closes"]`: close-price vector
- `payload["timestamps"]`: candle close timestamps
- keyword args `predlen`, `lookback`, `model_path`

Return formats supported by the adapter:

- single numeric score for `predlen=1`
- list of numeric scores for `predlen=16`
- dict with `steps=[...]`
- explicit step dicts like `{"direction": "UP", "score": 0.42}`

## Restart Behavior

The state file persists at least:

- last processed candle close time
- current regime
- open order ids
- last known balances
- stored 16-step forecast sequence
- current forecast index
- forecast cycle start timestamp

On restart the bot:

1. syncs server time
2. refreshes balances
3. cancels stale open orders for the symbol
4. restores any persisted `predlen=16` forecast buffer
5. catches up any missed closed candles from recent history

## Tests

Run the included unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Deployment Notes

- Use one Binance account or API key pair per bot to keep ownership clean.
- Run each script under a supervisor such as `systemd`, `supervisord`, or `tmux`.
- Review `runtime/logs/*.log` and `runtime/state/*.json` after first startup.
- Replace the stub model before enabling real money trading.
