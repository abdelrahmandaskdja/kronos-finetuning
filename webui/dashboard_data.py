from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


WEBUI_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEBUI_ROOT.parent


def _abs_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path_like: str | Path) -> dict[str, Any]:
    return json.loads(_abs_path(path_like).read_text())


def _read_csv(path_like: str | Path) -> pd.DataFrame:
    return pd.read_csv(_abs_path(path_like))


def _baseline(up_ratio: float) -> float:
    return max(up_ratio, 1.0 - up_ratio)


def _detect_time_column(df: pd.DataFrame) -> str:
    for column in ("timestamp", "timestamps", "timestamp_str", "date"):
        if column in df.columns:
            return column
    raise KeyError("No timestamp-like column found")


def _with_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    time_col = _detect_time_column(enriched)
    enriched["_timestamp"] = pd.to_datetime(enriched[time_col], errors="coerce")
    enriched = enriched.dropna(subset=["_timestamp"]).reset_index(drop=True)
    return enriched


def _format_dt(value: pd.Timestamp | str | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _iso(value: pd.Timestamp | str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).isoformat()


def _sample_frame(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.reset_index(drop=True)
    idx = pd.Series(range(limit)).apply(lambda i: round(i * (len(df) - 1) / (limit - 1)))
    return df.iloc[idx.unique()].reset_index(drop=True)


def _chart_payload(
    timestamps: list[str],
    series: list[dict[str, Any]],
    *,
    height: int = 240,
    compact: bool = False,
) -> dict[str, Any]:
    width = 960
    padding_left = 22 if compact else 38
    padding_right = 10
    padding_top = 14
    padding_bottom = 18 if compact else 28

    values: list[float] = []
    for item in series:
        for point in item["values"]:
            if point is None:
                continue
            try:
                values.append(float(point))
            except (TypeError, ValueError):
                continue

    if not values:
        values = [0.0, 1.0]

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    denom = max(y_max - y_min, 1e-9)
    point_count = max(len(timestamps), 1)
    x_denom = max(point_count - 1, 1)

    def _y_pos(value: float) -> float:
        ratio = (value - y_min) / denom
        return padding_top + (1.0 - ratio) * plot_height

    chart_series = []
    for item in series:
        points = []
        for idx, raw_value in enumerate(item["values"]):
            if raw_value is None:
                continue
            x = padding_left + (idx / x_denom) * plot_width
            y = _y_pos(float(raw_value))
            points.append((x, y))
        if not points:
            continue
        path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        chart_series.append(
            {
                "label": item["label"],
                "color": item["color"],
                "path": path,
            }
        )

    tick_count = 3 if compact else 4
    y_ticks = []
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        value = y_min + (y_max - y_min) * ratio
        y_ticks.append(
            {
                "label": f"{value:.1f}",
                "y": _y_pos(value),
            }
        )

    x_labels = []
    for idx in (0, point_count // 2, point_count - 1):
        x = padding_left + (idx / x_denom) * plot_width
        label = timestamps[idx] if timestamps else ""
        x_labels.append({"label": label, "x": x})

    return {
        "width": width,
        "height": height,
        "compact": compact,
        "series": chart_series,
        "y_ticks": y_ticks,
        "x_labels": x_labels,
    }


def _direction_badge(direction: int | bool | str) -> str:
    if isinstance(direction, str):
        return direction
    return "Up" if int(direction) == 1 else "Down"


@dataclass(frozen=True)
class ModelSpec:
    interval: str
    title: str
    family: str
    overview_note: str
    historical_summary: str
    historical_predictions: str | None = None
    historical_market: str | None = None
    live_summary: str | None = None
    live_predictions: str | None = None
    live_market: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "5min": ModelSpec(
        interval="5min",
        title="5-minute native direction head",
        family="direction_head",
        overview_note=(
            "Best saved short-horizon result in the repo, but the edge is still modest."
        ),
        historical_summary=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "oldsplit_5min_2026_to_2026-03-09_summary.json"
        ),
        historical_predictions=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "oldsplit_5min_2026_to_2026-03-09_predictions.csv"
        ),
        historical_market=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "BTCUSDT_kline_5min_context600_plus_2026-03-09.csv"
        ),
        live_summary=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "oldsplit_5min_2026_to_2026-03-09_summary.json"
        ),
        live_predictions=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "oldsplit_5min_2026_to_2026-03-09_predictions.csv"
        ),
        live_market=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "BTCUSDT_kline_5min_context600_plus_2026-03-09.csv"
        ),
    ),
    "15m": ModelSpec(
        interval="15m",
        title="15-minute consecutive rollout",
        family="rollout_ohlc",
        overview_note=(
            "This is the best saved 15-minute artifact I found locally, but it remains near "
            "baseline. The repo's stronger medium-horizon evidence is at 1h."
        ),
        historical_summary=(
            "finetune_csv/runs_requested_by_user/"
            "retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313/"
            "eval_epoch1_2026_to_2026-03-21_results/metrics_15m.json"
        ),
        historical_predictions=(
            "finetune_csv/runs_requested_by_user/"
            "retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313/"
            "eval_epoch1_2026_to_2026-03-21_results/pred_15m.csv"
        ),
        live_summary=(
            "finetune_csv/runs_requested_by_user/"
            "retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313/"
            "eval_epoch1_2026_to_2026-03-21_results/metrics_15m.json"
        ),
        live_predictions=(
            "finetune_csv/runs_requested_by_user/"
            "retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313/"
            "eval_epoch1_2026_to_2026-03-21_results/pred_15m.csv"
        ),
    ),
    "1d": ModelSpec(
        interval="1d",
        title="1-day last-6 path adapter",
        family="daily_path",
        overview_note=(
            "Strongest saved daily deployment result, with a visible generalization drop in "
            "later March refreshes."
        ),
        historical_summary=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "cons_only_lr3e6_last6_adapter_1d_close_to_close_fullrange_2026-03-16.json"
        ),
        historical_predictions=(
            "finetune_csv/runs_requested_by_user/conference_report_20260319/"
            "cons_only_lr3e6_last6_adapter_1d_close_to_close_fullrange_2026-03-16_rows.csv"
        ),
        live_summary=(
            "finetune_csv/runs_requested_by_user/"
            "eval_1d_2026_to_2026-03-29_cons_only_lr3e6_last6_adapter_metrics.json"
        ),
        live_predictions=(
            "finetune_csv/runs_requested_by_user/"
            "eval_1d_2026_to_2026-03-29_cons_only_lr3e6_last6_adapter_predictions.csv"
        ),
        live_market="finetune_csv/data/BTCUSDT_kline_1d_2026_to_2026-03-29_eval.csv",
    ),
}


def intervals() -> list[str]:
    return list(MODEL_SPECS.keys())


def _build_5min_overview() -> dict[str, Any]:
    spec = MODEL_SPECS["5min"]
    summary = _read_json(spec.historical_summary)
    accuracy = float(summary["trade_accuracy_all_rows"]) * 100.0
    baseline = float(summary["actual_up_ratio_all_rows"]) * 100.0
    delta = accuracy - baseline
    return {
        "interval": spec.interval,
        "title": spec.title,
        "family": "Direction head",
        "metric_label": "Trade accuracy",
        "accuracy_pct": accuracy,
        "baseline_pct": baseline,
        "delta_pp": delta,
        "rows": int(summary["trade_rows"]),
        "range_start": summary["range_start_actual"],
        "range_end": summary["range_end_actual"],
        "predicted_up_pct": float(summary["pred_up_ratio_all_rows"]) * 100.0,
        "actual_up_pct": float(summary["actual_up_ratio_all_rows"]) * 100.0,
        "pnl_usdt": float(summary["total_pnl_1btc_usdt"]),
        "return_pct": float(summary["compounded_return_pct_from_1.0"]),
        "headline": spec.overview_note,
        "analysis_points": [
            "Native 5-minute supervision outperformed the transferred variants on absolute PnL.",
            "The direction edge is positive, but only by a little more than one percentage point.",
            "Predicted-up ratio stayed below the actual-up ratio, which helped control long bias.",
        ],
        "source_paths": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
        ],
    }


def _build_15m_overview() -> dict[str, Any]:
    spec = MODEL_SPECS["15m"]
    summary = _read_json(spec.historical_summary)
    accuracy = float(summary["directional_accuracy_close_vs_prev_actual_close"]) * 100.0
    baseline = float(summary["majority_baseline_accuracy"]) * 100.0
    delta = accuracy - baseline
    return {
        "interval": spec.interval,
        "title": spec.title,
        "family": "Autoregressive rollout",
        "metric_label": "Close vs previous actual close",
        "accuracy_pct": accuracy,
        "baseline_pct": baseline,
        "delta_pp": delta,
        "rows": int(summary["rows"]),
        "range_start": summary["range_pred_start"],
        "range_end": summary["range_pred_end"],
        "predicted_up_pct": float(summary["predicted_up_ratio"]) * 100.0,
        "actual_up_pct": float(summary["actual_up_ratio"]) * 100.0,
        "pnl_usdt": None,
        "return_pct": None,
        "headline": spec.overview_note,
        "analysis_points": [
            "This 15-minute model is effectively flat against its baseline.",
            "Prediction bias leans slightly upward even though hit rate does not improve.",
            "I kept it here because you explicitly asked for 15m instead of the stronger repo-wide 1h result.",
        ],
        "source_paths": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
        ],
    }


def _build_1d_overview() -> dict[str, Any]:
    spec = MODEL_SPECS["1d"]
    summary = _read_json(spec.historical_summary)
    accuracy = float(summary["close_to_close_accuracy_consecutive_path"]) * 100.0
    baseline = _baseline(float(summary["actual_up_ratio_close_to_close"])) * 100.0
    delta = accuracy - baseline
    return {
        "interval": spec.interval,
        "title": spec.title,
        "family": "Path-consistent adapter",
        "metric_label": "Close-to-close consecutive path",
        "accuracy_pct": accuracy,
        "baseline_pct": baseline,
        "delta_pp": delta,
        "rows": int(summary["rows"]),
        "range_start": summary["range_start"],
        "range_end": summary["range_end"],
        "predicted_up_pct": float(summary["pred_up_ratio_close_to_close"]) * 100.0,
        "actual_up_pct": float(summary["actual_up_ratio_close_to_close"]) * 100.0,
        "pnl_usdt": float(summary["total_close_to_close_pnl_1btc_usdt"]),
        "return_pct": float(summary["compounded_close_to_close_return_pct_from_1.0"]),
        "headline": spec.overview_note,
        "analysis_points": [
            "Pure consecutive-path training with last-6 unfreezing was the strongest daily recipe.",
            "This is the largest direction edge and the largest PnL result in the requested set.",
            "Later March refreshes stayed above baseline but with a much smaller margin.",
        ],
        "source_paths": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
        ],
    }


@lru_cache(maxsize=1)
def get_overview_payload() -> dict[str, Any]:
    models = [
        _build_5min_overview(),
        _build_15m_overview(),
        _build_1d_overview(),
    ]
    comparison = [
        {
            "interval": model["interval"],
            "title": model["title"],
            "accuracy_label": _format_pct(model["accuracy_pct"]),
            "baseline_label": _format_pct(model["baseline_pct"]),
            "delta_label": f"{model['delta_pp']:+.2f} pp",
            "bias_label": (
                f"{model['predicted_up_pct']:.2f}% predicted up vs "
                f"{model['actual_up_pct']:.2f}% actual up"
            ),
            "pnl_label": _format_money(model["pnl_usdt"]),
        }
        for model in models
    ]
    return {
        "models": models,
        "comparison": comparison,
        "as_of_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "notes": [
            "Overview metrics come from the saved benchmark artifacts already present in the repo.",
            "The requested middle slot is 15m, even though the repo's strongest medium-horizon benchmark is 1h.",
            "The live page polls local files; it does not run fresh inference in the request path.",
        ],
    }


def _build_5min_detail() -> dict[str, Any]:
    spec = MODEL_SPECS["5min"]
    summary = _read_json(spec.historical_summary)
    df = _with_timestamp(_read_csv(spec.historical_predictions))
    valid = df.loc[df["mask"].fillna(0).astype(int) == 1].copy()
    valid["label"] = valid["label"].astype(int)
    valid["pred_dir"] = valid["pred_dir"].astype(int)
    valid["correct"] = (valid["label"] == valid["pred_dir"]).astype(int)
    valid["cumulative_accuracy_pct"] = valid["correct"].expanding().mean() * 100.0

    probability_slice = valid.tail(180).reset_index(drop=True)
    accuracy_slice = _sample_frame(valid[["_timestamp", "cumulative_accuracy_pct"]], 180)

    probability_chart = _chart_payload(
        [_format_dt(ts) for ts in probability_slice["_timestamp"]],
        [
            {
                "label": "Predicted up probability",
                "color": "#0f766e",
                "values": (probability_slice["pred_prob_up"] * 100.0).round(4).tolist(),
            },
            {
                "label": "Actual direction",
                "color": "#c67c2c",
                "values": (probability_slice["label"] * 100.0).tolist(),
            },
        ],
        height=250,
    )
    cumulative_chart = _chart_payload(
        [_format_dt(ts) for ts in accuracy_slice["_timestamp"]],
        [
            {
                "label": "Cumulative hit rate",
                "color": "#183153",
                "values": accuracy_slice["cumulative_accuracy_pct"].round(4).tolist(),
            }
        ],
        height=220,
    )

    tape = [
        {
            "timestamp": _format_dt(row["_timestamp"]),
            "predicted": _direction_badge(row["pred_dir"]),
            "actual": _direction_badge(row["label"]),
            "correct": bool(row["correct"]),
            "prob_up_pct": float(row["pred_prob_up"]) * 100.0,
        }
        for _, row in valid.tail(48).iterrows()
    ]

    return {
        "interval": spec.interval,
        "title": spec.title,
        "family_label": "Direction head",
        "metric_label": "Trade accuracy across all rows",
        "metric_value": float(summary["trade_accuracy_all_rows"]) * 100.0,
        "baseline_pct": float(summary["actual_up_ratio_all_rows"]) * 100.0,
        "delta_pp": (float(summary["trade_accuracy_all_rows"]) - float(summary["actual_up_ratio_all_rows"])) * 100.0,
        "rows": int(summary["trade_rows"]),
        "range_start": summary["range_start_actual"],
        "range_end": summary["range_end_actual"],
        "headline": "Best saved 5m deployment result, but not a dominant edge.",
        "summary_points": [
            "Primary benchmark uses trade accuracy on every row, not only masked evaluation rows.",
            "Masked classification accuracy is a little higher than the trade-accuracy headline.",
            "Native 5m supervision is the reason this model stayed competitive at the shortest horizon.",
        ],
        "stats": [
            {"label": "Trade accuracy", "value": _format_pct(float(summary["trade_accuracy_all_rows"]) * 100.0)},
            {"label": "Masked eval accuracy", "value": _format_pct(float(summary["directional_accuracy"]) * 100.0)},
            {"label": "Baseline", "value": _format_pct(float(summary["actual_up_ratio_all_rows"]) * 100.0)},
            {"label": "Total PnL", "value": _format_money(float(summary["total_pnl_1btc_usdt"]))},
            {"label": "Compounded return", "value": _format_pct(float(summary["compounded_return_pct_from_1.0"]))},
            {"label": "Predicted up ratio", "value": _format_pct(float(summary["pred_up_ratio_all_rows"]) * 100.0)},
        ],
        "charts": [
            {
                "title": "Last 180 Probability Decisions",
                "subtitle": "Green is the model's probability of an up move. Gold is the realized binary direction.",
                "payload": probability_chart,
            },
            {
                "title": "Cumulative Hit Rate",
                "subtitle": "Running accuracy on valid masked rows across the evaluation window.",
                "payload": cumulative_chart,
            },
        ],
        "tape_title": "Recent Prediction Tape",
        "tape_subtitle": "Last 48 valid signals with probability and realized direction.",
        "tape": tape,
        "table_columns": ["Timestamp", "Predicted", "Actual", "Prob Up", "Correct"],
        "table_rows": [
            [
                row["timestamp"],
                row["predicted"],
                row["actual"],
                _format_pct(row["prob_up_pct"]),
                "Yes" if row["correct"] else "No",
            ]
            for row in reversed(tape[-12:])
        ],
        "sources": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
            str(_abs_path(spec.historical_market)),
        ],
    }


def _build_15m_detail() -> dict[str, Any]:
    spec = MODEL_SPECS["15m"]
    summary = _read_json(spec.historical_summary)
    pred_df = _with_timestamp(_read_csv(spec.historical_predictions))
    actual_df = _with_timestamp(_read_csv(summary["data_path"]))

    merged = pred_df.merge(
        actual_df[["_timestamp", "open", "close"]],
        on="_timestamp",
        how="inner",
        suffixes=("_pred", "_actual"),
    )
    merged["pred_candle_dir"] = (merged["close_pred"] >= merged["open_pred"]).astype(int)
    merged["actual_candle_dir"] = (merged["close_actual"] >= merged["open_actual"]).astype(int)
    merged["abs_error"] = (merged["close_pred"] - merged["close_actual"]).abs()

    chart_slice = merged.tail(160).reset_index(drop=True)
    base_pred = chart_slice["close_pred"].iloc[0]
    base_actual = chart_slice["close_actual"].iloc[0]
    chart = _chart_payload(
        [_format_dt(ts) for ts in chart_slice["_timestamp"]],
        [
            {
                "label": "Predicted close (indexed)",
                "color": "#0f766e",
                "values": ((chart_slice["close_pred"] / base_pred) * 100.0).round(4).tolist(),
            },
            {
                "label": "Actual close (indexed)",
                "color": "#c67c2c",
                "values": ((chart_slice["close_actual"] / base_actual) * 100.0).round(4).tolist(),
            },
        ],
        height=250,
    )

    recent_rows = [
        [
            _format_dt(row["_timestamp"]),
            f"{row['close_pred']:,.2f}",
            f"{row['close_actual']:,.2f}",
            f"{row['abs_error']:,.2f}",
            _direction_badge(row["pred_candle_dir"]),
            _direction_badge(row["actual_candle_dir"]),
        ]
        for _, row in chart_slice.tail(12).iterrows()
    ]

    return {
        "interval": spec.interval,
        "title": spec.title,
        "family_label": "Autoregressive rollout",
        "metric_label": "Direction accuracy vs previous actual close",
        "metric_value": float(summary["directional_accuracy_close_vs_prev_actual_close"]) * 100.0,
        "baseline_pct": float(summary["majority_baseline_accuracy"]) * 100.0,
        "delta_pp": (
            float(summary["directional_accuracy_close_vs_prev_actual_close"])
            - float(summary["majority_baseline_accuracy"])
        )
        * 100.0,
        "rows": int(summary["rows"]),
        "range_start": summary["range_pred_start"],
        "range_end": summary["range_pred_end"],
        "headline": "Useful as a requested 15m slot, but it does not currently clear its baseline.",
        "summary_points": [
            "The model predicts full OHLC candles, which makes the close-path comparison easy to inspect.",
            "Hit rate is close to coin-flip, so the detailed view focuses on error behavior and path drift.",
            "If you later want the stronger medium-horizon story, the repo's benchmark winner is 1h, not 15m.",
        ],
        "stats": [
            {"label": "Accuracy", "value": _format_pct(float(summary["directional_accuracy_close_vs_prev_actual_close"]) * 100.0)},
            {"label": "Baseline", "value": _format_pct(float(summary["majority_baseline_accuracy"]) * 100.0)},
            {"label": "Close MAE", "value": _format_number(float(summary["regression"]["close_mae"]))},
            {"label": "Close RMSE", "value": _format_number(float(summary["regression"]["close_rmse"]))},
            {"label": "Predicted up ratio", "value": _format_pct(float(summary["predicted_up_ratio"]) * 100.0)},
            {"label": "Actual up ratio", "value": _format_pct(float(summary["actual_up_ratio"]) * 100.0)},
        ],
        "charts": [
            {
                "title": "Predicted vs Actual Close Path",
                "subtitle": "Last 160 rows, indexed to 100 at the start of the visible window.",
                "payload": chart,
            }
        ],
        "tape_title": "Latest 15m Rows",
        "tape_subtitle": "Recent predicted and realized closes with absolute error.",
        "tape": [],
        "table_columns": ["Timestamp", "Pred Close", "Actual Close", "Abs Error", "Pred Candle", "Actual Candle"],
        "table_rows": recent_rows,
        "sources": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
            str(_abs_path(summary["data_path"])),
        ],
    }


def _build_1d_detail() -> dict[str, Any]:
    spec = MODEL_SPECS["1d"]
    summary = _read_json(spec.historical_summary)
    rows_df = _with_timestamp(_read_csv(spec.historical_predictions))
    extension_df = pd.read_csv(
        REPO_ROOT / "reports/sgp2_extension_20260405/daily_winner_extension.csv"
    )

    close_chart = _chart_payload(
        [_format_dt(ts) for ts in rows_df["_timestamp"]],
        [
            {
                "label": "Predicted close",
                "color": "#0f766e",
                "values": rows_df["pred_close"].round(4).tolist(),
            },
            {
                "label": "Actual close",
                "color": "#c67c2c",
                "values": rows_df["actual_close"].round(4).tolist(),
            },
        ],
        height=260,
    )
    pnl_chart = _chart_payload(
        [_format_dt(ts) for ts in rows_df["_timestamp"]],
        [
            {
                "label": "Cumulative PnL",
                "color": "#183153",
                "values": rows_df["cum_close_to_close_pnl_1btc_usdt"].round(4).tolist(),
            }
        ],
        height=220,
    )
    extension_chart = _chart_payload(
        extension_df["window_end"].tolist(),
        [
            {
                "label": "Winner accuracy",
                "color": "#0f766e",
                "values": extension_df["path_accuracy_pct"].round(4).tolist(),
            },
            {
                "label": "Baseline",
                "color": "#c67c2c",
                "values": extension_df["baseline_pct"].round(4).tolist(),
            },
        ],
        height=210,
    )

    table_rows = [
        [
            _format_dt(row["_timestamp"]),
            _direction_badge(row["pred_close_to_close_direction"]),
            _direction_badge(row["actual_close_to_close_direction"]),
            "Yes" if int(row["direction_correct_close_to_close"]) == 1 else "No",
            _format_money(float(row["close_to_close_pnl_1btc_usdt"])),
            _format_money(float(row["cum_close_to_close_pnl_1btc_usdt"])),
        ]
        for _, row in rows_df.tail(12).iterrows()
    ]

    return {
        "interval": spec.interval,
        "title": spec.title,
        "family_label": "Path-consistent adapter",
        "metric_label": "Close-to-close consecutive path accuracy",
        "metric_value": float(summary["close_to_close_accuracy_consecutive_path"]) * 100.0,
        "baseline_pct": _baseline(float(summary["actual_up_ratio_close_to_close"])) * 100.0,
        "delta_pp": (
            float(summary["close_to_close_accuracy_consecutive_path"])
            - _baseline(float(summary["actual_up_ratio_close_to_close"]))
        )
        * 100.0,
        "rows": int(summary["rows"]),
        "range_start": summary["range_start"],
        "range_end": summary["range_end"],
        "headline": "Strongest daily result in the requested set, but later March drift matters.",
        "summary_points": [
            "This winner used pure path-consistent training with last-6 backbone blocks unfrozen.",
            "The raw deployment result is strong, but the later extension chart shows why the daily claim needs caution.",
            "The historical detail page therefore shows both the benchmark path and the later margin shrinkage.",
        ],
        "stats": [
            {"label": "Accuracy", "value": _format_pct(float(summary["close_to_close_accuracy_consecutive_path"]) * 100.0)},
            {"label": "Baseline", "value": _format_pct(_baseline(float(summary["actual_up_ratio_close_to_close"])) * 100.0)},
            {"label": "Total PnL", "value": _format_money(float(summary["total_close_to_close_pnl_1btc_usdt"]))},
            {"label": "Compounded return", "value": _format_pct(float(summary["compounded_close_to_close_return_pct_from_1.0"]))},
            {"label": "Predicted up ratio", "value": _format_pct(float(summary["pred_up_ratio_close_to_close"]) * 100.0)},
            {"label": "Actual up ratio", "value": _format_pct(float(summary["actual_up_ratio_close_to_close"]) * 100.0)},
        ],
        "charts": [
            {
                "title": "Predicted vs Actual Daily Close",
                "subtitle": "Full evaluation window from the saved March 16 conference snapshot.",
                "payload": close_chart,
            },
            {
                "title": "Cumulative Close-to-Close PnL",
                "subtitle": "1 BTC frictionless directional rule across the same rows.",
                "payload": pnl_chart,
            },
            {
                "title": "Later Generalization Check",
                "subtitle": "The daily winner stayed above baseline later, but by a much smaller margin.",
                "payload": extension_chart,
            },
        ],
        "tape_title": "Latest Daily Decisions",
        "tape_subtitle": "Last 12 benchmarked daily signals with realized close-to-close PnL.",
        "tape": [],
        "table_columns": ["Timestamp", "Predicted", "Actual", "Correct", "Row PnL", "Cum PnL"],
        "table_rows": table_rows,
        "sources": [
            str(_abs_path(spec.historical_summary)),
            str(_abs_path(spec.historical_predictions)),
            str(REPO_ROOT / "reports/sgp2_extension_20260405/daily_winner_extension.csv"),
        ],
    }


@lru_cache(maxsize=8)
def get_historical_detail(interval: str) -> dict[str, Any]:
    if interval == "5min":
        return _build_5min_detail()
    if interval == "15m":
        return _build_15m_detail()
    if interval == "1d":
        return _build_1d_detail()
    raise KeyError(interval)


def _latest_source_timestamp(path: str | Path | None) -> str:
    resolved = _abs_path(path)
    return resolved.name if resolved else "n/a"


def _build_live_5min() -> dict[str, Any]:
    spec = MODEL_SPECS["5min"]
    summary = _read_json(spec.live_summary)
    pred_df = _with_timestamp(_read_csv(spec.live_predictions))
    market_df = _with_timestamp(_read_csv(spec.live_market))

    latest_signal = pred_df.iloc[-1]
    latest_market = market_df.iloc[-1]
    spark_df = market_df.tail(72).reset_index(drop=True)

    sparkline = _chart_payload(
        [_format_dt(ts) for ts in spark_df["_timestamp"]],
        [{"label": "Close", "color": "#0f766e", "values": spark_df["close"].round(4).tolist()}],
        height=120,
        compact=True,
    )

    return {
        "interval": spec.interval,
        "title": spec.title,
        "family": "Direction head",
        "market_timestamp": _format_dt(latest_market["_timestamp"]),
        "signal_timestamp": _format_dt(latest_signal["_timestamp"]),
        "latest_close": float(latest_market["close"]),
        "signal_direction": _direction_badge(latest_signal["pred_dir"]),
        "signal_basis": "One-step direction head",
        "signal_strength": f"{abs(float(latest_signal['pred_prob_up']) - 0.5) * 200.0:.2f} pts from 50%",
        "summary_stat": _format_pct(float(summary["trade_accuracy_all_rows"]) * 100.0),
        "summary_label": "Saved trade accuracy",
        "market_source": str(_abs_path(spec.live_market)),
        "prediction_source": str(_abs_path(spec.live_predictions)),
        "sparkline": sparkline,
    }


def _build_live_15m() -> dict[str, Any]:
    spec = MODEL_SPECS["15m"]
    summary = _read_json(spec.live_summary)
    pred_df = _with_timestamp(_read_csv(spec.live_predictions))
    market_df = _with_timestamp(_read_csv(summary["data_path"]))

    latest_pred = pred_df.iloc[-1]
    latest_market = market_df.iloc[-1]
    spark_df = market_df.tail(72).reset_index(drop=True)
    direction = int(latest_pred["close"] >= latest_pred["open"])

    sparkline = _chart_payload(
        [_format_dt(ts) for ts in spark_df["_timestamp"]],
        [{"label": "Close", "color": "#183153", "values": spark_df["close"].round(4).tolist()}],
        height=120,
        compact=True,
    )

    return {
        "interval": spec.interval,
        "title": spec.title,
        "family": "Autoregressive rollout",
        "market_timestamp": _format_dt(latest_market["_timestamp"]),
        "signal_timestamp": _format_dt(latest_pred["_timestamp"]),
        "latest_close": float(latest_market["close"]),
        "signal_direction": _direction_badge(direction),
        "signal_basis": "Predicted candle close vs open",
        "signal_strength": f"{abs(float(latest_pred['close'] - latest_pred['open'])) / max(float(latest_pred['open']), 1e-9) * 100.0:.2f}% body",
        "summary_stat": _format_pct(float(summary["directional_accuracy_close_vs_prev_actual_close"]) * 100.0),
        "summary_label": "Saved 15m accuracy",
        "market_source": str(_abs_path(summary["data_path"])),
        "prediction_source": str(_abs_path(spec.live_predictions)),
        "sparkline": sparkline,
    }


def _build_live_1d() -> dict[str, Any]:
    spec = MODEL_SPECS["1d"]
    summary = _read_json(spec.live_summary)
    pred_df = _with_timestamp(_read_csv(spec.live_predictions))
    market_df = _with_timestamp(_read_csv(spec.live_market))

    latest_pred = pred_df.iloc[-1]
    latest_market = market_df.iloc[-1]
    spark_df = market_df.tail(60).reset_index(drop=True)
    direction = int(latest_pred["close"] >= latest_pred["open"])

    sparkline = _chart_payload(
        [_format_dt(ts) for ts in spark_df["_timestamp"]],
        [{"label": "Close", "color": "#c67c2c", "values": spark_df["close"].round(4).tolist()}],
        height=120,
        compact=True,
    )

    return {
        "interval": spec.interval,
        "title": f"{spec.title} live slice",
        "family": "Path-consistent adapter",
        "market_timestamp": _format_dt(latest_market["_timestamp"]),
        "signal_timestamp": _format_dt(latest_pred["_timestamp"]),
        "latest_close": float(latest_market["close"]),
        "signal_direction": _direction_badge(direction),
        "signal_basis": "Predicted candle close vs open",
        "signal_strength": f"{abs(float(latest_pred['close'] - latest_pred['open'])) / max(float(latest_pred['open']), 1e-9) * 100.0:.2f}% body",
        "summary_stat": _format_pct(float(summary["directional_accuracy_close_vs_prev_close_consecutive_path"]) * 100.0),
        "summary_label": "Latest saved path accuracy",
        "market_source": str(_abs_path(spec.live_market)),
        "prediction_source": str(_abs_path(spec.live_predictions)),
        "sparkline": sparkline,
    }


def get_live_payload() -> dict[str, Any]:
    snapshots = [
        _build_live_5min(),
        _build_live_15m(),
        _build_live_1d(),
    ]
    return {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "snapshots": snapshots,
        "notes": [
            "This page polls local artifact files that already exist in the repo.",
            "If upstream jobs refresh those CSV or JSON files, the cards update without reloading the page.",
            "The live page surfaces the freshest saved local signal for each requested interval. It does not launch inference jobs on refresh.",
        ],
    }
