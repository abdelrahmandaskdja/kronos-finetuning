from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import PredictionResult


class PredictionStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_move_pct REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def log_prediction(self, result: PredictionResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO prediction_logs (
                    id, created_at, symbol, horizon, model_id, signal, confidence, predicted_move_pct, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.generated_at,
                    result.symbol,
                    result.horizon,
                    result.model_id,
                    result.signal,
                    result.confidence,
                    result.predicted_move_pct,
                    json.dumps(result.model_dump(), ensure_ascii=True),
                ),
            )
            connection.commit()

    def list_predictions(self, limit: int = 30) -> list[PredictionResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM prediction_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [PredictionResult.model_validate_json(row["payload_json"]) for row in rows]

    def latest_prediction(
        self,
        symbol: str | None = None,
        horizon: str | None = None,
        model_id: str | None = None,
    ) -> PredictionResult | None:
        clauses: list[str] = []
        values: list[str] = []
        if symbol:
            clauses.append("symbol = ?")
            values.append(symbol)
        if horizon:
            clauses.append("horizon = ?")
            values.append(horizon)
        if model_id:
            clauses.append("model_id = ?")
            values.append(model_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT payload_json
            FROM prediction_logs
            {where}
            ORDER BY created_at DESC
            LIMIT 1
        """
        with self._connect() as connection:
            row = connection.execute(query, tuple(values)).fetchone()
        if not row:
            return None
        return PredictionResult.model_validate_json(row["payload_json"])

