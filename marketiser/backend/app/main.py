from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .config import SETTINGS
from .db import PredictionStore
from .services.market_data import MarketDataService
from .services.model_registry import ModelRegistry
from .services.prediction_engine import PredictionEngine
from .schemas import PredictionRequest
from .websocket_manager import WebSocketManager


async def seed_default_predictions(app: FastAPI) -> None:
    del app


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = SETTINGS
    ws_manager = WebSocketManager()
    store = PredictionStore(settings.database_path)
    store.init()
    model_registry = ModelRegistry(settings)
    await model_registry.load()
    market_data = MarketDataService(settings, ws_manager)
    await market_data.start()
    prediction_engine = PredictionEngine(model_registry, market_data, store, ws_manager)

    app.state.settings = settings
    app.state.ws_manager = ws_manager
    app.state.store = store
    app.state.model_registry = model_registry
    app.state.market_data = market_data
    app.state.prediction_engine = prediction_engine
    await seed_default_predictions(app)
    yield
    await market_data.stop()


app = FastAPI(
    title=SETTINGS.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=SETTINGS.api_prefix)


@app.websocket("/ws/terminal")
async def terminal_stream(websocket: WebSocket) -> None:
    ws_manager: WebSocketManager = app.state.ws_manager
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json(
            {
                "type": "bootstrap",
                "payload": {
                    "status": app.state.market_data.get_status().model_dump(),
                    "watchlist": [item.model_dump() for item in app.state.market_data.get_watchlist()],
                    "models": [item.model_dump() for item in app.state.model_registry.list_models()],
                    "logs": [
                        item.model_dump()
                        for item in app.state.prediction_engine.list_logs(limit=20, symbols=app.state.settings.symbols)
                    ],
                    "families": [item.model_dump() for item in app.state.model_registry.families()],
                },
            }
        )
        while True:
            message = await websocket.receive_text()
            if message.lower() == "ping":
                await websocket.send_json({"type": "pong", "payload": {}})
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)
