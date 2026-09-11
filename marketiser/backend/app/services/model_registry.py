from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..config import Kronos_ROOT, Settings
from ..schemas import BenchmarkSummary, ModelFamilyRecord, ModelRecord


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics file {path} did not contain a JSON object")
    return payload


def _nested_get(payload: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _window_from_metrics(payload: dict[str, Any]) -> str | None:
    start = payload.get("range_pred_start") or payload.get("range_start") or payload.get("range_start_actual")
    end = payload.get("range_pred_end") or payload.get("range_end") or payload.get("range_end_actual")
    if start and end:
        return f"{start} to {end}"
    eval_ts = payload.get("eval_timestamp_utc")
    return _string_or_none(eval_ts)


def _ratio_from_pct(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return number / 100.0


def _report_row(path: Path, *, model_key: str, interval: str) -> dict[str, Any]:
    payload = _json_load(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Report file {path} did not contain a rows list")
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("model_key") == model_key and row.get("interval") == interval:
            return row
    raise KeyError(f"Could not find report row for model_key={model_key} interval={interval} in {path}")


def _csv_report_row(path: Path, *, interval_label: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("interval_label") == interval_label:
                return row
    raise KeyError(f"Could not find CSV report row for interval_label={interval_label} in {path}")


def _benchmark_from_metrics(
    path: Path,
    *,
    accuracy_key: str | None = None,
    close_direction_key: str | None = None,
    path_direction_key: str | None = None,
    candle_direction_key: str | None = None,
    majority_key: str | None = None,
    lookback_key: str | None = "lookback",
    pred_len_key: str | None = "pred_len",
    close_mae_key: str | None = None,
    winner_metric_name: str | None = None,
    winner_metric_key: str | None = None,
    selection_summary: str | None = None,
) -> BenchmarkSummary:
    payload = _json_load(path)
    return BenchmarkSummary(
        accuracy=_float_or_none(_nested_get(payload, accuracy_key)),
        evaluation_window=_window_from_metrics(payload),
        lookback=_int_or_none(_nested_get(payload, lookback_key)),
        pred_len=_int_or_none(_nested_get(payload, pred_len_key)),
        close_direction_accuracy=_float_or_none(_nested_get(payload, close_direction_key or accuracy_key)),
        path_direction_accuracy=_float_or_none(_nested_get(payload, path_direction_key)),
        candle_direction_accuracy=_float_or_none(_nested_get(payload, candle_direction_key)),
        majority_baseline_accuracy=_float_or_none(_nested_get(payload, majority_key)),
        close_mae=_float_or_none(_nested_get(payload, close_mae_key)),
        winner_metric_name=winner_metric_name,
        winner_metric_value=_float_or_none(_nested_get(payload, winner_metric_key)),
        selection_summary=selection_summary,
    )


def _runtime_from_metrics(
    path: Path,
    *,
    pred_len_key: str | None = "pred_len",
    block_len_key: str | None = "block_len",
) -> dict[str, Any]:
    payload = _json_load(path)
    pred_len = _int_or_none(_nested_get(payload, pred_len_key))
    if pred_len is None:
        pred_len = _int_or_none(_nested_get(payload, block_len_key))
    return {
        "model_id": _string_or_none(payload.get("model_id")),
        "tokenizer_id": _string_or_none(payload.get("tokenizer_id")),
        "lookback": _int_or_none(payload.get("lookback")),
        "pred_len": pred_len,
    }


def _benchmark_from_report_row(
    row: dict[str, Any],
    raw_metrics_path: Path,
    *,
    winner_metric_name: str,
    winner_metric_key: str,
    selection_summary: str,
) -> BenchmarkSummary:
    raw = _json_load(raw_metrics_path)
    return BenchmarkSummary(
        accuracy=_ratio_from_pct(row.get("accuracy_all_rows_pct")),
        evaluation_window=f"{row['window_start']} to {row['window_end']}",
        lookback=_int_or_none(raw.get("lookback_window")),
        pred_len=_int_or_none(raw.get("horizon_steps")),
        close_direction_accuracy=_ratio_from_pct(row.get("accuracy_all_rows_pct")),
        majority_baseline_accuracy=_ratio_from_pct(row.get("baseline_all_rows_pct")),
        winner_metric_name=winner_metric_name,
        winner_metric_value=_ratio_from_pct(row.get(winner_metric_key)),
        selection_summary=selection_summary,
    )


def _benchmark_with_summary_row(
    benchmark: BenchmarkSummary,
    row: dict[str, Any],
    *,
    winner_metric_name: str,
) -> BenchmarkSummary:
    return benchmark.model_copy(
        update={
            "accuracy": _ratio_from_pct(row.get("accuracy_pct")),
            "close_direction_accuracy": _ratio_from_pct(row.get("accuracy_pct")),
            "majority_baseline_accuracy": _ratio_from_pct(row.get("baseline_pct")),
            "evaluation_window": f"{row['range_start']} to {row['range_end']}",
            "pnl_usdt": _float_or_none(row.get("total_pnl_usdt")),
            "compounded_return_pct": _float_or_none(row.get("compounded_return_pct")),
            "winner_metric_name": winner_metric_name,
            "winner_metric_value": _ratio_from_pct(row.get("accuracy_pct")),
            "selection_summary": row.get("rationale"),
        }
    )


def _foundation_cache_dir(model_id: str) -> Path:
    owner, name = model_id.split("/", maxsplit=1)
    return Kronos_ROOT / ".hf_cache" / "hub" / f"models--{owner}--{name}"


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, ModelRecord] = {}
        self._families: list[ModelFamilyRecord] = [
            ModelFamilyRecord(
                key="kronos_foundation",
                title="Kronos Foundation Models",
                summary="Direct backbone inference from the public Kronos model zoo with configurable lookback and prediction length.",
                strengths=["Real Kronos runtime", "Base / small / mini switching", "Configurable lookback and pred_len"],
            ),
            ModelFamilyRecord(
                key="native_direction_head",
                title="Finetuned 5m Kronos",
                summary="Real finetuned Kronos checkpoint for the 5-minute branch with its benchmark metrics preserved alongside live runtime access.",
                strengths=["Live finetuned runtime", "5-minute winner", "Benchmark metrics preserved"],
            ),
            ModelFamilyRecord(
                key="pruned_transfer_head",
                title="Finetuned 1h Direction Head",
                summary="Hourly pruned50 Kronos direction-head winner wired for live inference with its report-selected benchmark metrics preserved for side-by-side inspection.",
                strengths=["Live hourly runtime", "Pruned50 direction head", "Report winner metrics preserved"],
            ),
            ModelFamilyRecord(
                key="consecutive_path_adapter",
                title="Finetuned 1d Kronos",
                summary="Daily finetuned Kronos checkpoint with rollout-style benchmark metrics and live inference support.",
                strengths=["Path forecasting", "Daily finetuned runtime", "Saved rollout metrics"],
            ),
        ]

    async def load(self) -> None:
        selected_winners_summary = (
            Kronos_ROOT
            / "reports"
            / "sgp2_extension_20260405"
            / "selected_winners_summary.csv"
        )
        selected_5m_row = _csv_report_row(selected_winners_summary, interval_label="5min")
        selected_1d_row = _csv_report_row(selected_winners_summary, interval_label="1d")

        metrics_5m = Kronos_ROOT / "finetune_csv" / "eval_watch_predictor_2026" / "eval_20260217_155238.json"
        runtime_5m = _runtime_from_metrics(metrics_5m)

        report_1h_summary = (
            Kronos_ROOT
            / "reports"
            / "ad_hoc_20260409"
            / "cross_timeframe_direction_heads_2026-04-08"
            / "cross_timeframe_accuracy_summary.json"
        )
        metrics_1h = (
            Kronos_ROOT
            / "reports"
            / "ad_hoc_20260409"
            / "cross_timeframe_direction_heads_2026-04-08"
            / "raw"
            / "1h_pruned50_winner"
            / "1h"
            / "raw_metrics.json"
        )
        report_1h_row = _report_row(report_1h_summary, model_key="1h_pruned50_winner", interval="1h")
        runtime_1h_payload = _json_load(metrics_1h)
        checkpoint_1h = Path(str(runtime_1h_payload["checkpoint_path"]))

        metrics_1d = (
            Kronos_ROOT
            / "finetune_csv"
            / "runs_requested_by_user"
            / "eval_1d_2026-01-01_to_2026-04-05_cons_only_lr3e6_last6_adapter"
            / "metrics.json"
        )
        runtime_1d = _runtime_from_metrics(metrics_1d)

        definitions = [
            {
                "id": "kronos-mini",
                "display_name": "Kronos-mini",
                "scope": "foundation",
                "family": "kronos_foundation",
                "branch": "foundation-mini",
                "supported_horizons": list(self.settings.forecast_horizons),
                "params": "4.1M",
                "checkpoint_path": None,
                "runtime_model_id": "NeoQuasar/Kronos-mini",
                "runtime_tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
                "context_length": 2048,
                "default_lookback": 512,
                "default_pred_len": 16,
                "supports_live_inference": True,
                "supports_custom_lookback": True,
                "supports_custom_pred_len": True,
                "supports_path_forecast": True,
                "execution_mode": "kronos-runtime",
                "description": "Foundation Kronos mini model for fast live inference. Best when you want the shortest turnaround with a larger context ceiling.",
                "benchmark": None,
            },
            {
                "id": "kronos-small",
                "display_name": "Kronos-small",
                "scope": "foundation",
                "family": "kronos_foundation",
                "branch": "foundation-small",
                "supported_horizons": list(self.settings.forecast_horizons),
                "params": "24.7M",
                "checkpoint_path": None,
                "runtime_model_id": "NeoQuasar/Kronos-small",
                "runtime_tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
                "context_length": 512,
                "default_lookback": 256,
                "default_pred_len": 16,
                "supports_live_inference": True,
                "supports_custom_lookback": True,
                "supports_custom_pred_len": True,
                "supports_path_forecast": True,
                "execution_mode": "kronos-runtime",
                "description": "Balanced foundation Kronos model for live path forecasting with moderate latency and the standard 512-token context budget.",
                "benchmark": None,
            },
            {
                "id": "kronos-base",
                "display_name": "Kronos-base",
                "scope": "foundation",
                "family": "kronos_foundation",
                "branch": "foundation-base",
                "supported_horizons": list(self.settings.forecast_horizons),
                "params": "102.3M",
                "checkpoint_path": str(_foundation_cache_dir("NeoQuasar/Kronos-base")),
                "runtime_model_id": "NeoQuasar/Kronos-base",
                "runtime_tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
                "context_length": 512,
                "default_lookback": 256,
                "default_pred_len": 16,
                "supports_live_inference": True,
                "supports_custom_lookback": True,
                "supports_custom_pred_len": True,
                "supports_path_forecast": True,
                "execution_mode": "kronos-runtime",
                "description": "Largest public Kronos foundation model in this repo. Use it when you want the strongest backbone at the cost of slower live inference.",
                "benchmark": None,
            },
            {
                "id": "finetuned-5m-native",
                "display_name": "BTCUSDT 5m Finetuned Winner",
                "scope": "finetuned",
                "family": "native_direction_head",
                "branch": "5m-winner",
                "supported_horizons": ["5m"],
                "params": "102.3M ft",
                "checkpoint_path": runtime_5m["model_id"],
                "runtime_model_id": runtime_5m["model_id"],
                "runtime_tokenizer_id": runtime_5m["tokenizer_id"],
                "context_length": 512,
                "default_lookback": runtime_5m["lookback"] or 512,
                "default_pred_len": runtime_5m["pred_len"] or 1,
                "supports_live_inference": True,
                "supports_custom_lookback": False,
                "supports_custom_pred_len": False,
                "supports_path_forecast": True,
                "execution_mode": "kronos-finetuned",
                "description": "Finetuned 5-minute Kronos checkpoint with fixed training defaults. Live runs use the saved finetuned weights while the benchmark cards keep its recorded evaluation metrics.",
                "benchmark": _benchmark_with_summary_row(
                    _benchmark_from_metrics(
                        metrics_5m,
                        accuracy_key="directional_accuracy_close_vs_last_close",
                        close_direction_key="directional_accuracy_close_vs_last_close",
                        candle_direction_key="directional_accuracy_candle_close_vs_open",
                        majority_key="majority_baseline_accuracy",
                        winner_metric_name="2026 close direction accuracy",
                        winner_metric_key="directional_accuracy_close_vs_last_close",
                        selection_summary="Preserved as the current 5-minute finetuned checkpoint from the repo winner snapshot.",
                    ),
                    selected_5m_row,
                    winner_metric_name="Report-selected 5m close direction accuracy",
                ),
            },
            {
                "id": "finetuned-1h-pruned",
                "display_name": "BTCUSDT 1h Pruned50 Winner",
                "scope": "finetuned",
                "family": "pruned_transfer_head",
                "branch": "1h-pruned50-winner",
                "supported_horizons": ["1h"],
                "params": "102.3M dir-head",
                "checkpoint_path": str(checkpoint_1h),
                "runtime_model_id": None,
                "runtime_tokenizer_id": None,
                "context_length": 512,
                "default_lookback": _int_or_none(runtime_1h_payload.get("lookback_window")) or 512,
                "default_pred_len": _int_or_none(runtime_1h_payload.get("horizon_steps")) or 1,
                "supports_live_inference": True,
                "supports_custom_lookback": False,
                "supports_custom_pred_len": False,
                "supports_path_forecast": False,
                "execution_mode": "kronos-direction-head",
                "description": "Hourly pruned50 Kronos direction-head winner from the cross-timeframe report. Live runs execute the saved direction-head checkpoint and the benchmark cards preserve the report-selected 1-hour metrics.",
                "benchmark": _benchmark_from_report_row(
                    report_1h_row,
                    metrics_1h,
                    winner_metric_name="Cross-timeframe 1h direction accuracy",
                    winner_metric_key="accuracy_all_rows_pct",
                    selection_summary="Swapped to the report-selected 1h pruned50 direction-head winner because it led the preserved 1-hour sweep at 53.27% all-rows accuracy.",
                ),
            },
            {
                "id": "finetuned-1d-path",
                "display_name": "BTCUSDT 1d Finetuned Winner",
                "scope": "finetuned",
                "family": "consecutive_path_adapter",
                "branch": "1d-winner",
                "supported_horizons": ["1d"],
                "params": "102.3M ft",
                "checkpoint_path": runtime_1d["model_id"],
                "runtime_model_id": runtime_1d["model_id"],
                "runtime_tokenizer_id": runtime_1d["tokenizer_id"],
                "context_length": 256,
                "default_lookback": runtime_1d["lookback"] or 256,
                "default_pred_len": runtime_1d["pred_len"] or 16,
                "supports_live_inference": True,
                "supports_custom_lookback": False,
                "supports_custom_pred_len": False,
                "supports_path_forecast": True,
                "execution_mode": "kronos-finetuned",
                "description": "Daily finetuned Kronos checkpoint with fixed training defaults. Live runs use the finetuned model while the benchmark cards preserve the rollout evaluation metrics.",
                "benchmark": _benchmark_with_summary_row(
                    _benchmark_from_metrics(
                        metrics_1d,
                        accuracy_key="directional_accuracy_close_vs_prev_actual_close",
                        close_direction_key="directional_accuracy_close_vs_prev_actual_close",
                        path_direction_key="directional_accuracy_close_vs_prev_close_consecutive_path",
                        candle_direction_key="directional_accuracy_candle_close_vs_open",
                        majority_key="majority_baseline_accuracy",
                        pred_len_key="block_len",
                        close_mae_key="regression.close_mae",
                        winner_metric_name="2026 consecutive path accuracy",
                        winner_metric_key="directional_accuracy_close_vs_prev_close_consecutive_path",
                        selection_summary="Kept as the daily finetuned winner because it led the saved daily sweep on consecutive path accuracy.",
                    ),
                    selected_1d_row,
                    winner_metric_name="Report-selected 1d close direction accuracy",
                ),
            },
        ]

        models: dict[str, ModelRecord] = {}
        for definition in definitions:
            checkpoint_path = definition["checkpoint_path"]
            runtime_model_id = definition["runtime_model_id"]
            runtime_tokenizer_id = definition["runtime_tokenizer_id"]
            status = "loaded"

            if checkpoint_path and not Path(checkpoint_path).exists():
                status = "unavailable"
            if runtime_tokenizer_id:
                tokenizer_path = Path(runtime_tokenizer_id)
                if tokenizer_path.is_absolute() and not tokenizer_path.exists():
                    status = "unavailable"
            if definition["scope"] == "foundation":
                if runtime_model_id and runtime_tokenizer_id:
                    model_cached = _foundation_cache_dir(runtime_model_id).exists()
                    tokenizer_cached = _foundation_cache_dir(runtime_tokenizer_id).exists()
                    status = "loaded" if model_cached and tokenizer_cached else "loading"

            models[definition["id"]] = ModelRecord(
                id=definition["id"],
                display_name=definition["display_name"],
                scope=definition["scope"],
                family=definition["family"],
                branch=definition["branch"],
                supported_assets=list(self.settings.symbols),
                supported_horizons=list(definition["supported_horizons"]),
                status=status,
                params=definition["params"],
                description=definition["description"],
                execution_mode=definition["execution_mode"],
                checkpoint_path=_string_or_none(checkpoint_path),
                runtime_model_id=_string_or_none(runtime_model_id),
                runtime_tokenizer_id=_string_or_none(runtime_tokenizer_id),
                context_length=definition["context_length"],
                default_lookback=definition["default_lookback"],
                default_pred_len=definition["default_pred_len"],
                supports_live_inference=definition["supports_live_inference"],
                supports_custom_lookback=definition["supports_custom_lookback"],
                supports_custom_pred_len=definition["supports_custom_pred_len"],
                supports_path_forecast=definition["supports_path_forecast"],
                benchmark=definition["benchmark"],
                tags=["kronos", definition["scope"], definition["branch"]],
            )
        self._models = models

    def list_models(self, symbol: str | None = None, horizon: str | None = None) -> list[ModelRecord]:
        del symbol
        models = list(self._models.values())
        if horizon:
            models = [model for model in models if horizon in model.supported_horizons]
        return models

    def get_model(self, model_id: str) -> ModelRecord:
        return self._models[model_id]

    def families(self) -> list[ModelFamilyRecord]:
        return list(self._families)

    def count(self) -> int:
        return len(self._models)
