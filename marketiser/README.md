# Marketiser

Marketiser is a live BTCUSDT forecasting and analytics platform built around Kronos-style research models.

This MVP is intentionally modular:

- `frontend/`
  Next.js + React + TypeScript + Tailwind terminal UI
- `backend/`
  FastAPI + WebSocket market stream + model registry + prediction service

## Architecture

### Frontend

- `app/page.tsx`
  Premium landing page with Marketiser brand and CTA into the terminal
- `app/terminal/page.tsx`
  Core trading terminal route
- `components/terminal/*`
  Top bar, watchlist, backend-driven chart panel, signal sidebar, bottom research tabs
- `hooks/use-marketiser-stream.ts`
  Optional WebSocket subscription layer for incremental updates on top of REST polling
- `store/terminal-store.ts`
  Zustand terminal state for active symbol, interval, horizon, models, logs, candles, and live status

### Backend

- `app/services/market_data.py`
  Binance REST polling manager with reconnect logic and mock streaming fallback
- `app/services/model_registry.py`
  Startup-loaded model registry for the 5-minute, 1-hour, and 1-day branches
- `app/services/prediction_engine.py`
  Prediction interface plus immediate mock-safe inference driver
- `app/db.py`
  SQLite-backed prediction logging
- `app/api/routes.py`
  REST API surface for watchlist, models, predictions, and system summary
- `app/main.py`
  FastAPI entrypoint plus `/ws/terminal` WebSocket gateway

## Live product behavior

- Live market data is polled from Binance REST when available.
- The current deployment path is intentionally narrowed to `BTCUSDT` to keep the terminal responsive while the single-asset experience is hardened.
- If Binance REST is unavailable, the backend falls back to simulated market updates so the terminal still runs.
- The frontend now talks to the backend through a same-origin Next.js proxy route by default, which avoids brittle browser-side hardcoded backend URLs.
- Model registry auto-loads on startup.
- Prediction logs persist into SQLite.
- The main terminal chart now renders from Marketiser candle data directly, with prediction-path overlays driven by backend responses.

## Current model branches

- `kronos-native-5m`
  Native direction-head model for the `5m` branch
- `kronos-pruned-1h`
  Pruned transfer direction-head model for the `1h` branch
- `kronos-path-1d`
  Consecutive-path adapter model for the `1d` branch
- `kronos-research-general`
  Placeholder slot for future custom models and research comparisons

## Project structure

```text
marketiser/
  README.md
  docker-compose.yml
  backend/
    .env.example
    Dockerfile
    requirements.txt
    app/
      api/
      services/
      config.py
      db.py
      main.py
      schemas.py
      websocket_manager.py
  frontend/
    .env.example
    Dockerfile
    app/
    components/
    hooks/
    lib/
    store/
```

## Local setup

### 1. Backend

```bash
cd marketiser/backend
python3 -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If your Python environment is PEP 668 managed, install with either a virtualenv or:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

### 2. Frontend

```bash
cd marketiser/frontend
npm install
cp .env.example .env.local
npm run dev
```

Then open `http://localhost:3000`.

## Environment templates

### Frontend

- `MARKETISER_BACKEND_URL`
- `NEXT_PUBLIC_WS_URL` optional

### Backend

- `MARKETISER_ENABLE_BINANCE_REST`
- `MARKETISER_USE_MOCK_PREDICTIONS`
- `MARKETISER_SYMBOLS`
- `MARKETISER_CHART_INTERVALS`
- `MARKETISER_FORECAST_HORIZONS`
- `MARKETISER_BINANCE_REST_URL`
- `MARKETISER_BINANCE_SNAPSHOT_POLL_SECONDS`
- `MARKETISER_BINANCE_CANDLE_POLL_SECONDS`

## REST endpoints

- `GET /api/health`
- `GET /api/system/summary`
- `GET /api/watchlist`
- `GET /api/market/status`
- `GET /api/market/candles`
- `GET /api/models`
- `GET /api/model-families`
- `GET /api/predictions/logs`
- `GET /api/predictions/latest`
- `POST /api/predictions/run`

## WebSocket

- `ws://localhost:8000/ws/terminal`

The frontend no longer depends on a direct browser WebSocket to function. It uses REST polling by default through the Next.js proxy route, and `NEXT_PUBLIC_WS_URL` only adds optional push updates.

Events emitted:

- `bootstrap`
- `system.status`
- `market.snapshot`
- `market.kline`
- `prediction.created`

## Research note

Marketiser is a research and analytics platform. It is not financial advice.

## Future-ready placeholders

- News and sentiment fusion
- Model comparison mode
- Leaderboard of best models
- Saved layouts
- Alert engine
- Broker / execution integration
