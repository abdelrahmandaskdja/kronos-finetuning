# Multi-Interval Binance Spot Bot

This bot combines the five interval-specific models selected from the repo's 2025 and 2026 reports:

- `1d`: `BTCUSDT_kline_1m_2017_2024_ce_consdir_leakfix`
- `4h`: `BTCUSDT_kline_1m_trainall_2025H1_biasctrl_epoch6_lowest_ce`
- `1h`: `BTCUSDT_kline_1m_trainall_2025H1_biasctrl`
- `15m`: `5min native direction head`
- `1m`: `BTCUSDT_kline_1m_2017_2024_ce_consdir_leakfix`

## Strategy

The pipeline is intentionally long-only and spot-only:

- Signals are synchronized on the `15m` decision clock.
- Consecutive models use the same tested direction metric as the reports: `pred_close_vs_prev_actual_close`.
- The `15m` direction-head uses the same tested metric as the reports: `pred_dir_vs_next_bar_label`.
- `1d` and `4h` define regime. `15m` is the decision trigger. `1h` and `1m` are confirmations.
- When `1d`, `4h`, and `15m` align, the strategy builds a signed target exposure and scales it up if `1h` and `1m` agree.
- On Binance Spot, negative exposure cannot be executed as a real short, so bearish synchronized signals flatten to cash instead of matching the report's short-side PnL.
- If a required model (`1d`, `4h`, or `15m`) fails to generate a live signal, the bot refuses to trade for that cycle and logs the blocking error. If an optional layer (`1h` or `1m`) fails, it is treated as neutral for sizing.
- If Binance REST returns too few recent bars for the strict tested lookback, the bot falls back to the freshest local 2026 market CSV available for that interval.

This is the closest live spot deployment policy to the saved test metric. Exact report-style PnL parity would require a futures long/short execution adapter.

## Execution

The default execution style is now passive maker-limit:

- The bot uses Binance Spot `LIMIT_MAKER` orders, which are post-only. If a submitted price would immediately trade as a taker, Binance rejects the order instead of filling it aggressively.
- For buys, it joins the best bid when the spread is only one tick wide and steps one tick inside the spread when there is room.
- For sells, it joins the best ask when the spread is only one tick wide and steps one tick inside the spread when there is room.
- Working orders created by this bot use the `kronosmkr` client-order-id prefix. Only those managed orders are canceled or replaced; manual account orders are left alone.
- If the target drops to flat, the bot can cancel its own stale maker orders even when no new trade is needed.
- When polling faster than the `15m` signal clock, the bot can keep repricing its own working maker order inside the same bar instead of waiting for the next signal bar.

This aims to reduce taker fees and spread crossing, but it does not guarantee zero fees. Actual maker and taker commissions still depend on your Binance account tier and any BNB fee discount settings.

For the standalone `1m` strategy, the ready-made configs now enable a faster live path:

- The decision clock is aligned to Binance server time instead of a coarse local polling interval.
- The bot maintains the current `1m` candle from Binance real-time trade events rather than waiting for the slower `1m` kline-close stream.
- New maker entries can go through Binance Spot WebSocket API `order.place` and use `LIMIT_MAKER` with `pegPriceType=PRIMARY_PEG`, which posts at the best price on the same side of the book at the matching engine.
- The fast loop can automatically use CUDA for the `1m` model when available and prewarms the model plus stability profile before the first live minute close.
- The `BTCFDUSD` standalone `1m` config can also consume the model's full 16-step forecast block and only refresh inference after the block is exhausted, which avoids overlapping one-minute re-rollouts.
- This is a latency reduction path, not a hard guarantee. Network jitter, exchange-side throttling, and matching-engine response time still apply.

## Safety Defaults

- Default mode is `dry_run`.
- No signed order is sent unless `--send-orders` is provided.
- Even in `testnet` or `live` mode, the script only previews the rebalance unless `--send-orders` is enabled.
- Default order style is `maker_limit`. If you want the old immediate execution path, set `"execution": {"order_style": "market"}` in a config overlay.
- The default config caps a single rebalance at `250 USDT` notional.
- The script enforces a daily and intraday drawdown halt and will flatten by targeting `0%` allocation when a halt is triggered.
- Orders are evaluated on new `15m` primary-bar closes unless `--force` is used.

## Usage

Dry-run once:

```bash
python trading/binance_spot_multi_interval_bot.py --mode dry_run
```

Dry-run continuously:

```bash
python trading/binance_spot_multi_interval_bot.py --mode dry_run --loop --poll-seconds 60
```

Testnet preview with account data:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode testnet --env-file trading/secrets/testnet.env
```

Demo Mode preview with account data:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode demo
```

Testnet signed order validation only:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode testnet --send-orders --test-order
```

Demo Mode signed order validation only:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode demo --send-orders --test-order
```

Actual testnet orders:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode testnet --send-orders
```

Actual Demo Mode orders:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode demo --send-orders
```

Live spot orders:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
python trading/binance_spot_multi_interval_bot.py --mode live --send-orders --env-file trading/secrets/live.env
```

Live spot orders with the old market-order behavior:

```bash
python trading/binance_spot_multi_interval_bot.py --mode live --send-orders --config my_market_overlay.json
```

## Config Overlay

You can override defaults with a JSON file:

```json
{
  "risk": {
    "max_trade_notional_usdt": 100.0,
    "min_rebalance_notional_usdt": 20.0
  },
  "execution": {
    "order_style": "maker_limit",
    "maker_inside_spread_ticks": 1,
    "maker_reprice_after_ticks": 1,
    "maker_buy_balance_buffer_pct": 0.25,
    "allow_same_bar_order_maintenance": true
  }
}
```

Run it with:

```bash
python trading/binance_spot_multi_interval_bot.py --config my_bot_config.json
```

Ready-made standalone configs:

- `trading/configs/strategy_15m_only.json`: trades only the `15m` direction-head model
- `trading/configs/strategy_1m_only.json`: trades only the `1m` consecutive model

Examples:

```bash
python trading/binance_spot_multi_interval_bot.py --mode demo --send-orders --config trading/configs/strategy_15m_only.json
python trading/binance_spot_multi_interval_bot.py --mode demo --send-orders --config trading/configs/strategy_1m_only.json
```

Always-on `1m` standalone loop:

```bash
python trading/binance_spot_multi_interval_bot.py --mode demo --send-orders --loop --poll-seconds 5 --config trading/configs/strategy_1m_only.json
```

Fast live `1m` standalone loop:

```bash
python trading/binance_spot_multi_interval_bot.py --mode live --send-orders --loop --config trading/configs/strategy_1m_only_btcfdusd.json --env-file trading/secrets/1m.env
```

The loop now prints a concise heartbeat each cycle and writes a separate `status_log.jsonl` alongside `signal_log.jsonl` and `order_log.jsonl`.

## Notes

- This bot uses Binance Spot, not futures. Bearish signals flatten to cash; they do not open short positions.
- `testnet` uses `https://testnet.binance.vision`, while `demo` uses Binance Spot Demo Mode at `https://demo-api.binance.com`.
- The standalone strategy configs use separate runtime folders and separate client-order-id prefixes so they do not reuse each other's state or managed maker orders.
- The standalone runtime folders now also keep a `status_log.jsonl` file with per-cycle status, recent fill summaries, and current open-order counts.
- The `BTCFDUSD` standalone configs can each point at different API env vars. The ready-made live names are `BINANCE_1M_API_KEY`, `BINANCE_1M_API_SECRET`, `BINANCE_15M_API_KEY`, `BINANCE_15M_API_SECRET`, `BINANCE_1H_API_KEY`, `BINANCE_1H_API_SECRET`, `BINANCE_4H_API_KEY`, `BINANCE_4H_API_SECRET`, `BINANCE_1D_API_KEY`, and `BINANCE_1D_API_SECRET`.
- Market data is fetched from Binance REST klines each cycle and written under `trading/runtime/market_cache/`.
- The fast `1m` loop is only enabled for `live` and `testnet` modes because it depends on Binance market-data and trading WebSocket endpoints.
- Runtime logs and state are written under `trading/runtime/`.
- Dry-run now also previews the planned maker order using the live public book ticker when available.
- If you want symmetric long/short execution, the next step should be a separate USD-M futures adapter with its own risk limits.
